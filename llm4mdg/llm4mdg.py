import concurrent.futures
import hashlib
import os.path
import re
from itertools import chain
from string import ascii_letters, digits
from typing import List, Union

import yaml
from langchain.globals import set_verbose, set_debug
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr, validator, root_validator
from nanoid import generate as nanoid_gen

from .actions import *
from .actions.find_data_interactions import output_analysis
from .constant import CONFIG_LOC, NANOID_LENGTH, MULTI_THREAD_COUNT
from .logs import logger
from .models import *
from .models.data_interaction_models import NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis
from .models.deploy_config_models import DeployConfig, KubernetesDeployConfig, DockerComposeDeployConfig
from .utils import save_intermediate_result, load_intermediate_result, tree_with_root_dir_name, relative_path, \
    absolute_path, is_valid_key_in_dict, multi_thread, is_valid_string


class LLM4MDG(BaseModel):
    # Fields set by configuration file
    llm: BaseChat = Field(alias='chat_model')
    embd: OpenAIEmbedding = Field(alias='openai_embedding')
    vector_db: BaseVecDB
    graph_db: Neo4jGraphDB = Field(alias='neo4j')
    project_loc: str = Field(alias='project_location')
    config_center_name: str | None = None
    config_center_dir: str | None = None

    # Fields set on instantiation (Not visible when dumping model)
    process_id: str | None = Field(
        default_factory=lambda: nanoid_gen(size=NANOID_LENGTH, alphabet=ascii_letters + digits + "_"))
    _steps: int = PrivateAttr(default=0)

    def __init__(self, **kwargs):
        """Show banner when instantiating."""
        super().__init__(**kwargs)
        self._show_banner()

        # Set up vector store
        self.vector_db.embd_func = self.embd.m
        if self.vector_db.collection_name is None or self.vector_db.collection_name == "":
            self.vector_db.collection_name = f"vectordb_{self.process_id}"
        self.vector_db.init_db()
        logger.info("Vector store initialized successfully.")

        # Set up graph database (neo4j)
        self.graph_db.init_db()
        self.graph_db.collection_name = f"graphdb_{self.process_id}"
        logger.info("Graph database initialized successfully.")

    def __del__(self):
        self.graph_db.close_db()
        logger.info("Graph database connection closed successfully.")

    @staticmethod
    def from_config(file_path: str = CONFIG_LOC, **kwargs) -> 'LLM4MDG':
        """
        Initialize a new LLM4MDG instance from a configuration file.

        :param file_path: The path to the configuration file.
        :return: A new LLM4MDG instance.
        """
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        if config_data.get("debug"):
            set_debug(True)
            set_verbose(True)

        ret = LLM4MDG(**config_data, **kwargs)
        logger.info("<blue>Parsed LLM4MDG config:</blue>")
        logger.info(f"\t<yellow>Current running ID:</yellow> {ret.process_id}")
        logger.info(f"\t<yellow>Target:</yellow> {ret.project_loc}")
        logger.info(f"\t<yellow>Models:</yellow> CHAT({ret.llm.model.value}), EMBD({ret.embd.model.value})")
        logger.info(
            f"\t<yellow>VectorDB:</yellow> {ret.vector_db.db_type} (Connection: {ret.vector_db.connection_type})")
        logger.info(
            f"\t<yellow>GraphDB:</yellow> {ret.graph_db.uri}")

        # Show continue prompt
        # logger.info(f"<magenta><b>Proceed analysis? (Y/n)</b></magenta>")
        # choice = input()
        # if choice.lower() == "y" or choice.lower() == "":
        #     return ret
        # else:
        #     logger.info(f"User cancelled analysis, exiting.")
        #     exit(0)
        return ret

    @validator('llm', pre=True, always=True)
    @classmethod
    def set_llm(cls, value) -> BaseChat:
        assert (is_valid_key_in_dict(value, "type")
                and value.get('type') in ["openai", "anthropic",
                                          "google"]), f"Unsupported chat model type: {value.get('type')}"

        if value.get('type') == 'openai':
            return OpenAIChat(**value)
        elif value.get('type') == 'anthropic':
            return AnthropicChat(**value)
        elif value.get('type') == 'google':
            return GoogleChat(**value)

    @validator('vector_db', pre=True, always=True)
    @classmethod
    def set_vector_db(cls, value) -> BaseVecDB:
        assert value.get('db_type') in ["milvus", "chroma"], f"Unsupported vector_db type: {value.get('db_type')}"
        if value.get('db_type') == 'milvus':
            return MilvusVecDB(**value)
        elif value.get('db_type') == 'chroma':
            return ChromaVecDB(**value)

    @validator('process_id', pre=True, always=True)
    @classmethod
    def check_nanoid(cls, value: str) -> str:
        """
        Validator function checking `process_id`.
        """
        assert re.fullmatch("[_0-9a-zA-Z]{" + str(NANOID_LENGTH) + "}", value) is not None, "Not a valid NanoID."
        return value

    @staticmethod
    def _show_banner():
        logger.info(r"""<red>

        __    __    __  _____ __  __  _______  ______
       / /   / /   /  |/  / // / /  |/  / __ \/ ____/
      / /   / /   / /|_/ / // /_/ /|_/ / / / / / __
     / /___/ /___/ /  / /__  __/ /  / / /_/ / /_/ /
    /_____/_____/_/  /_/  /_/ /_/  /_/_____/\____/

</red>""")

    @staticmethod
    def _output_services(
            result: ValidatedResult,
            config_center: str = None
    ):
        logger.info(f"<yellow>Modifications from LLM:</yellow> {result.modification}")
        logger.info("<yellow>Deploy configuration files observed from project:</yellow>")
        for config in result.validated_result.deploy_config:
            logger.info(f"  <magenta>{config.path} (type: {config.type.value.upper()})</magenta>")
        logger.info("<yellow>Service instances observed from project: </yellow>")
        for service in result.validated_result.services:
            fmt_str = f"  <magenta>{service.name}</magenta>"
            if not service.prebuilt:
                fmt_str += f" <green>({service.source_dir})</green>"
            if service.name == config_center:
                fmt_str += f" [MARKED_AS_CONFIG_CENTER]"
            logger.info(fmt_str)

    def _print_new_stage(self, msg: str):
        self._steps += 1
        logger.info(f"<blue>Stage {self._steps}. {msg}</blue>")

    def _embd_data_id(self, key: str, service: str) -> str:
        msg = f"[{self.process_id}]_[{key}]_[{service}]"
        return hashlib.sha1(msg.encode()).hexdigest()

    def _is_config_center(self, service: Union[str, IdentifiedService]) -> bool:
        if isinstance(service, IdentifiedService):
            return self.config_center_name == service.name
        elif isinstance(service, str):
            return self.config_center_name == service
        else:
            return False

    def _identify_service(
            self,
            use_intermediate_result: bool = False,
    ) -> ValidatedResult:
        def _run_action():
            identified_services = IdentifyServiceAction(llm=self.llm, project_loc=self.project_loc).run()
            save_intermediate_result(
                "identify_service_action", self.process_id, identified_services.json(indent=4))
            return identified_services

        if use_intermediate_result:
            logger.info(f"Skip agent running and use intermediate result: identify_service_action_{self.process_id}.")
            try:
                return ValidatedResult.parse_raw(
                    load_intermediate_result("identify_service_action", self.process_id))
            except FileNotFoundError:
                return _run_action()
        else:
            return _run_action()

    def _find_config_center(
            self,
            result: ValidatedResult
    ):
        """
        Find instance name of configuration center service from given name or directory.
        """
        if (
                (self.config_center_dir is None and self.config_center_name is None) or
                (self.config_center_dir == "" and self.config_center_name == "")
        ):
            logger.debug("No config center specified, skipping.")
            return None

        name_matched, dir_matched = False, False
        config_center_name = ""
        for service in result.validated_result.services:
            if self.config_center_name == service.name:
                name_matched = True
                config_center_name = service.name
            if self.config_center_dir == service.source_dir:
                dir_matched = True
                config_center_name = service.name

        if dir_matched or name_matched:
            logger.debug(f"Found config center by name/directory: {config_center_name}.")
            return config_center_name
        else:
            logger.debug("No config center found, skipping.")
            return None

    def _process_config_center(
            self,
            services: IdentifyServiceResult,
            config_center_name: str,
            config_center_dir: str | None = None,
            use_intermediate_result: bool = False,
    ) -> ProcessConfigCenterResult:
        def _run_action():
            config_result = ProcessConfigCenterAction(
                llm=self.llm, identified_result=services,
                config_center_name=config_center_name,
                config_center_dir=config_center_dir,
                project_loc=self.project_loc).run()
            save_intermediate_result(
                "process_config_center_action", self.process_id, config_result.json(indent=4))
            return config_result

        if use_intermediate_result:
            logger.info(f"Skip analyzing and use intermediate result: process_config_center_action_{self.process_id}.")
            try:
                return ProcessConfigCenterResult.parse_raw(
                    load_intermediate_result("process_config_center_action", self.process_id))
            except FileNotFoundError:
                return _run_action()
        else:
            return _run_action()

    def _merge_config_from_config_center(
            self,
            services: IdentifyServiceResult,
            result: ProcessConfigCenterResult
    ) -> IdentifyServiceResult:
        for swc in result.services_with_configs:
            for service in services.services:
                if service.name == swc:
                    # Found a matching item
                    service.configs = result.services_with_configs[swc]
                    for i in range(len(service.configs)):
                        config_center_abs_path = absolute_path(self.project_loc, self.config_center_dir)
                        config_abs_path = os.path.abspath(os.path.join(config_center_abs_path, service.configs[i]))
                        service.configs[i] = "." + config_abs_path.removeprefix(self.project_loc)
                    break
        return services

    def _parse_deploy_configs(
            self,
            deploy_configs: List[DeployConfig],
            use_intermediate_result: bool = False
    ) -> List[DeployConfig]:
        def _run_action():
            parsed_configs = ParseDeployConfigsAction(
                project_loc=self.project_loc, deploy_configs=deploy_configs).run()

            _out = DeployConfigList(__root__=parsed_configs)
            save_intermediate_result(
                "parse_deploy_configs_action", self.process_id, _out.json(indent=4))
            return parsed_configs

        if use_intermediate_result:
            logger.info(f"Skip analyzing and use intermediate result: parse_deploy_configs_action_{self.process_id}.")
            try:
                # Problems here
                return DeployConfigList.parse_raw(
                    load_intermediate_result("parse_deploy_configs_action", self.process_id)).__root__
            except FileNotFoundError:
                return _run_action()
        else:
            return _run_action()

    def _embed_codes(
            self,
            service: IdentifiedService,
            public_configs: List[str] = None,
            use_intermediate_result: bool = False
    ):
        dir_str, dir_files = tree_with_root_dir_name(
            absolute_path(self.project_loc, service.source_dir), service.source_dir)

        # Exclude public config files from the embedding of config center
        if self._is_config_center(service) and public_configs is not None:
            for c in public_configs:
                try:
                    dir_files.remove(absolute_path(self.project_loc, c))
                except ValueError:
                    continue

        total_file_count = len(dir_files) if service.configs is None else len(dir_files) + len(service.configs)
        file_count_fmt = (f"<yellow>Processing non-prebuilt service</yellow> "
                          f"<magenta>{service.name}</magenta>, ")
        file_count_fmt += f"file counts: {total_file_count}"
        if service.configs is not None and len(service.configs) != 0:
            file_count_fmt += f" ({len(service.configs)} additional configs)"

        logger.info(file_count_fmt)

        # Check if this service has additional configs from config center.
        # If yes, add config files to file list and interpret them.
        if service.configs is not None and len(service.configs) > 0:
            dir_files.extend(map(lambda x: absolute_path(self.project_loc, x), service.configs))

        def process_file(file, project_loc, vector_db, _service, llm, _dir_str, _use_intermediate_result):
            rel_path = relative_path(project_loc, file)
            _file_data_hash = self._embd_data_id(rel_path, _service.name)

            if vector_db.get_data_count({"id": _file_data_hash}) == 1:
                if _use_intermediate_result:
                    logger.debug(f"Skip analyzing {rel_path}, service: {_service.name}, data_hash: {_file_data_hash}")
                    return None
                else:
                    logger.debug(
                        f"Overwrite data of {rel_path}, service: {_service.name}, data_hash: {_file_data_hash}")
                    vector_db.delete_data([_file_data_hash])

            code_content, code_interpretation = InterpretCodeAction(
                llm=llm, project_loc=project_loc,
                code_path=file, dir_structure=_dir_str,
                service_relative_path=_service.source_dir, additional_configs=_service.configs).run()

            if code_interpretation is None:
                logger.debug(f"Skip analyzing code {rel_path}, file is empty")
                return None

            _metadata = {
                "filepath": rel_path,
                "service_name": _service.name,
                "code_content": code_content,
            }

            return code_interpretation, _metadata, _file_data_hash

        # Deduplication
        dir_files = list(set(dir_files))
        _ret = multi_thread(
            func=process_file,
            data=dir_files,
            arg_name_of_data="file",
            thread_cnt=len(dir_files) if len(dir_files) < MULTI_THREAD_COUNT else MULTI_THREAD_COUNT,
            project_loc=self.project_loc,
            vector_db=self.vector_db,
            _service=service,
            llm=self.llm,
            _dir_str=dir_str,
            _use_intermediate_result=use_intermediate_result,
        )
        datas, metadatas, ids = \
            [i[0] for i in _ret], [i[1] for i in _ret], [i[2] for i in _ret]

        if len(datas) != 0 and len(metadatas) != 0 and len(ids) != 0:
            ret = self.vector_db.add_data(datas=datas, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully embedded {len(ret)} code files.")

    def _find_data_interactions(
            self,
            data: IdentifyServiceResult,
            use_intermediate_result: bool = False
    ) -> List[Union[NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis]]:
        def _run_action():
            data_interactions_result = FindDataInteractionsAction(
                llm=self.llm,
                vector_db=self.vector_db,
                project_loc=self.project_loc,
                configs=data.deploy_config,
                services=data.services,
            ).run()

            _out = ServiceAnalysisList(__root__=data_interactions_result)
            save_intermediate_result(
                "find_data_interactions_action", self.process_id, _out.json(indent=4))
            return data_interactions_result

        if use_intermediate_result:
            logger.info(f"Skip analyzing and use intermediate result: find_data_interactions_action_{self.process_id}.")

            try:
                _ret = ServiceAnalysisList.parse_raw(
                    load_intermediate_result("find_data_interactions_action", self.process_id)).__root__
                for data in _ret:
                    output_analysis(data)
                return _ret
            except FileNotFoundError:
                return _run_action()
        else:
            return _run_action()

    def _build_dependency_graph(
            self,
            services: IdentifyServiceResult,
            data_interactions: List[Union[NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis]],
            use_intermediate_result: bool = False
    ):
        if not use_intermediate_result:
            self.graph_db.reset_collection()
            self.graph_db.init_collection()
            BuildDependencyGraphAction(
                services=services,
                data_interactions=data_interactions,
                graph_db=self.graph_db,
            ).run()
        else:
            pass

    def run(self):
        """
        Start running the whole procedure.

        Stage 1. Identify service instances in project. (`self._identify_service()`)
        Stage 2. Process configuration files in config center service if they exist. (`self._process_config_center()`)
        Stage 3. Parse data in deploy configuration files. (`self._parse_deploy_configs()`)
        Stage 4. Embed codes and config files of non-prebuilt services into vector DB. (`self._process_non_prebuilt_service()`)
        Stage 5. Find data interactions in all services.
        Stage 6. Build microservice dependency graph.
        """
        logger.info("Start scanning target project.")

        self._print_new_stage("Identify service instances in project.")
        identified_result = self._identify_service(use_intermediate_result=True)
        logger.info("Service identification process completed.")

        config_center = self._find_config_center(identified_result)
        self._output_services(identified_result, config_center)
        identified_result = identified_result.validated_result

        if config_center is not None:
            self._print_new_stage("Process configuration files in config center service if they exist.")
            config_result = self._process_config_center(
                identified_result, config_center, self.config_center_dir,
                use_intermediate_result=True)

            logger.info(f"Grabbed config files from configuration center service, store type: {config_result.store}.")
            logger.info(f"<yellow>LLM analysis of config center service: </yellow>{config_result.analysis}")
            logger.info(f"Corrected faults in paths of configuration files. [SKIPPED]")

            # TODO: How to handle incorrect relative path of config files?
            identified_result = self._merge_config_from_config_center(identified_result, config_result)
            logger.info("Merged configuration files into corresponding services.")

        self._print_new_stage("Parse data in deploy configuration files.")
        parsed_configs = self._parse_deploy_configs(identified_result.deploy_config, use_intermediate_result=False)
        if len(parsed_configs) != 0:
            identified_result.deploy_config = parsed_configs
        logger.info("Parsed all environment variables and ports data from deploy configs.")

        self._print_new_stage("Embed codes and config files of non-prebuilt services into vector DB.")
        non_prebuilt_services = [s for s in identified_result.services if s.prebuilt is False]

        _public_configs = list(chain(*[s.configs for s in identified_result.services if s.configs is not None]))
        _public_configs = list(set(_public_configs))

        for service in non_prebuilt_services:
            self._embed_codes(
                service,
                public_configs=_public_configs if self._is_config_center(service) else None,
                use_intermediate_result=True
            )
        logger.info("Embedding of all non-prebuilt services' files is done.")

        self._print_new_stage("Find data interactions in all services.")
        data_interactions_result = self._find_data_interactions(identified_result, use_intermediate_result=True)
        logger.info("Data interactions between all services have been analyzed.")

        # self._print_new_stage("Build microservice dependency graph.")
        # self._build_dependency_graph(identified_result, data_interactions_result, use_intermediate_result=False)

        logger.info("<magenta>All stages completed successfully.</magenta>")


# Wrapper classes for dumping data.
class DeployConfigList(BaseModel):
    __root__: List[DeployConfig]


class ServiceAnalysisList(BaseModel):
    __root__: List[Union[NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis]]
