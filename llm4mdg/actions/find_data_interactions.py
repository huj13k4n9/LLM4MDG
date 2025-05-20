from typing import List, Union

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel

from .base import Action
from .summarize_content import SummarizeContentAction
from .tools import call_llm_and_return_result
from ..constant import MULTI_THREAD_COUNT
from ..logs import logger
from ..models import BaseVecDB, IdentifiedService, BaseChat
from ..models.data_interaction_models import PrebuiltServiceAnalysis, NonPrebuiltServiceAnalysis
from ..models.deploy_config_models import DeployConfig, DockerComposeDeployConfig, KubernetesDeployConfig
from ..models.deploy_config_models.kubernetes import KubernetesMetadata
from ..prompts import AnalyzePrebuiltServicePrompt, AnalyzeNonPrebuiltServicePrompt, QueryVectorDBPrompt, \
    ValidateDataInteractionsPrompt
from ..utils import multi_thread

_format_rag_str = '-' * 16 + "\nFILENAME: `{filename}`\nBRIEF:\n```\n{brief}\n```\n"


def output_analysis(analysis: Union[PrebuiltServiceAnalysis, NonPrebuiltServiceAnalysis]):
    logger.debug(f"{analysis.service_name}: {analysis}".replace('<', '\\<'))
    _fmt_str = []
    if analysis.service is not None:
        _fmt_str.append(f"<yellow>Service:</yellow> {analysis.service}")
    if analysis.type is not None:
        _fmt_str.append(f"<yellow>Type:</yellow> {analysis.type}")

    logger.info("<yellow>LLM analysis of prebuilt service</yellow> "
                f"<magenta>{analysis.service_name}</magenta>:")
    logger.info("  " + ", ".join(_fmt_str))

    if analysis.ports is not None and len(analysis.ports) != 0:
        _ports_str = []
        for p in analysis.ports:
            if "protocol" in p and p.get("protocol") is not None:
                _ports_str.append(f"{p.get('port')}({p.get('protocol')})")
            else:
                _ports_str.append(f"{p.get('port')}")
        logger.info(f"  <yellow>Open Ports:</yellow> {', '.join(_ports_str)}")

    if isinstance(analysis, NonPrebuiltServiceAnalysis):
        if analysis.language is not None and len(analysis.language) != 0:
            logger.info(f"  <yellow>Used language:</yellow> {', '.join(analysis.language)}")

        if analysis.interactions is not None and len(analysis.interactions) != 0:
            logger.info(f"  <yellow>Observed data interactions:</yellow>")
            for i in analysis.interactions:
                # To prevent unrecognized tags in input string, which causes an Exception in loguru.
                _escaped = str(i).replace('<', '\\<')
                logger.info(f"  - {_escaped}")

    _analysis = analysis.analysis.replace('<', '\\<')
    logger.info(f"  <yellow>Analysis message:</yellow> {_analysis}")


def _summarize(llm, data):
    summarized_content = SummarizeContentAction(
        llm=llm,
        brief="This text includes summarization of key code files and "
              "configuration files in a microservice project.",
        content=data,
        key_topics="\n".join([
            "1. Details about passively exposed interfaces to external services, "
            "including but not limited to HTTP APIs, gRPC interfaces, controllers, "
            "routes, listening ports, etc. If found, list relevant details like URI, "
            "host, port, request method, etc.",
            "2. Details about proactively interaction with any external "
            "services, including but not limited to consuming HTTP APIs or gRPC "
            "interfaces, connecting to specific external ports, invoking SOAP services, "
            "using message queues, etc. If found, list relevant details like URI, host, "
            "port, request method, etc.",
            "3. Basic definition and functionality of each file."
        ]),
    ).run()
    return summarized_content


def _format_rag_data(data):
    context = ""
    for result in data:
        if isinstance(result, list):
            for r in result:
                if isinstance(r, Document):
                    context += _format_rag_str.format(
                        filename=r.metadata.get('filepath'),
                        brief=r.page_content)
        elif isinstance(result, Document):
            context += _format_rag_str.format(
                filename=result.metadata.get('filepath'),
                brief=result.page_content)

    context += "-" * 16
    return context


class FindDataInteractionsAction(Action, BaseModel):
    llm: BaseChat
    vector_db: BaseVecDB
    project_loc: str
    configs: List[DeployConfig]
    services: List[IdentifiedService]
    iter_times: int = 30

    def _find_data_in_configs(self, service_name: str, attr: str):
        _ret = []
        for c in self.configs:
            if isinstance(c, DockerComposeDeployConfig):
                for deployment in [d for d in c.docker_deployments if d.name == service_name]:
                    v = getattr(deployment, attr, None)
                    if v is not None:
                        _ret.append(v)
            elif isinstance(c, KubernetesDeployConfig):
                # Services
                for service in [s for s in c.k8s_services if s.metadata.name_matched(service_name)]:
                    v = getattr(service, attr, None)
                    if v is not None:
                        _ret.append(v)
                # Pods
                for container in ([cc for d in c.k8s_pods
                                   if d.metadata.name_matched(service_name)
                                   for cc in d.containers] +
                                  [cc for d in c.k8s_deployments
                                   if d.metadata.name_matched(service_name)
                                   for cc in d.pod_template.containers]):
                    v = getattr(container, attr, None)
                    if v is not None:
                        _ret.append(v)
        return _ret

    def _fetch_rag_data(self, service_name: str):
        _filter = {"service_name": service_name}

        file_count = self.vector_db.get_data_count(filter_condition=_filter)
        logger.debug(f"Retrieved file count of {service_name}: {file_count}")

        retrieved_data = self.vector_db.retrieve_data(
            query=QueryVectorDBPrompt.get_prompt(),
            top_k=file_count,
            filter_condition=_filter,
            search_type="mmr",
        )

        _configs = [s.configs for s in self.services if s.name == service_name]
        for cfgs in _configs:
            if cfgs is not None:
                for c in cfgs:
                    _paths = []
                    for item in retrieved_data:
                        if isinstance(item, list):
                            _paths.extend([i.metadata.get("filepath") for i in item])
                        elif isinstance(item, Document):
                            _paths.append(item.metadata.get("filepath"))

                    if c not in _paths:
                        if self.vector_db.get_data_count({**_filter, "filepath": c}) != 0:
                            _retrieved_config = self.vector_db.retrieve_data(
                                query="",
                                top_k=1,
                                filter_condition={**_filter, "filepath": c},
                                search_type="mmr",
                            )
                            retrieved_data.append(_retrieved_config)

        return retrieved_data

    def run(self) -> List[Union[NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis]]:
        def _process_service(_s):
            logger.info(f"Analyzing service <magenta>{_s.name}</magenta>")
            _ports = [str(item) for sublist in self._find_data_in_configs(_s.name, "ports") for item in sublist]

            if _s.prebuilt:
                _images = self._find_data_in_configs(_s.name, "image")

                prompt = AnalyzePrebuiltServicePrompt.get_prompt(image_name=_images, ports=_ports)
                result = call_llm_and_return_result(self.llm, prompt, PrebuiltServiceAnalysis)
            else:
                rag_result = self._fetch_rag_data(_s.name)

                # TODO: Use consumed tokens instead of file count
                segment_size = 20
                if len(rag_result) > segment_size:
                    logger.info("RAG result too long, summarize it.")
                    segments = [_format_rag_data(rag_result[i:i + segment_size])
                                for i in range(0, len(rag_result), segment_size)]

                    _summarized = multi_thread(
                        func=_summarize,
                        data=segments,
                        arg_name_of_data="data",
                        thread_cnt=len(segments),
                        llm=self.llm,
                    )
                    rag_result = ("\n" + "-" * 16 + "\n").join(_summarized)
                else:
                    rag_result = _format_rag_data(rag_result)

                prompt = AnalyzeNonPrebuiltServicePrompt.get_prompt(
                    service_name=_s.name,
                    ports=_ports,
                    rag_result=rag_result,
                )
                result = call_llm_and_return_result(self.llm, prompt, NonPrebuiltServiceAnalysis)

                logger.info(f"Got analysis result, validating.")
                validate_prompt = ValidateDataInteractionsPrompt.get_prompt(
                    service_name=_s.name,
                    result=result.json()
                )
                result = call_llm_and_return_result(
                    self.llm, validate_prompt, NonPrebuiltServiceAnalysis)
                logger.info("Successfully got validated result.")

            result.service_name = _s.name
            output_analysis(result)
            return result

        _ret = multi_thread(
            func=_process_service,
            data=self.services,
            arg_name_of_data="_s",
            thread_cnt=len(self.services) if len(self.services) < MULTI_THREAD_COUNT else MULTI_THREAD_COUNT,
        )
        return _ret
