from typing import List, Union

from langchain_core.pydantic_v1 import BaseModel

from llm4mdg.actions.base import Action
from ..models import IdentifyServiceResult, Neo4jGraphDB
from ..models.data_interaction_models import NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis, DataInteractionType, \
    DataInteraction
from ..utils import is_valid_key_in_dict


class BuildDependencyGraphAction(Action, BaseModel):
    services: IdentifyServiceResult
    data_interactions: List[Union[NonPrebuiltServiceAnalysis, PrebuiltServiceAnalysis]]
    graph_db: Neo4jGraphDB

    @staticmethod
    def _find_interaction_by_port(interaction: List[DataInteraction], port: int | str):
        _ret = []

        for i in interaction:
            if is_valid_key_in_dict(i.interaction_details, "port"):
                _port = i.interaction_details.get("port")
                if str(port) == str(_port):
                    _ret.append(i)
            else:
                # Sometimes no host and port identified...
                i.interaction_details["port"] = port
                _ret.append(i)
        return _ret if len(_ret) > 0 else None

    def run(self):
        _qb = self.graph_db.qb

        # 1. create node for every service.
        for service in self.services.services:
            _data_interaction = [i for i in self.data_interactions if i.service_name == service.name]
            assert len(_data_interaction) == 1
            _data_interaction = _data_interaction[0]

            _args = self.graph_db.get_node_args("s", "Service", props={
                "name": service.name,
                "description": _data_interaction.service,
                "type": _data_interaction.type,
            })

            # MATCH (p:Project {...props})
            # CREATE (s:Service {...props})-[:BELONGS_TO]->(p)
            self.graph_db.run_statement(str(_qb.match().node(**self.graph_db.project_node)
                                            .create().node(**_args).related_to("BELONGS_TO").node(ref_name="p")))

            # 2. Parse data interactions.
            # 2.1. Add passively exposed interfaces.
            def _add_interface(port_info, kwargs=None):
                _interface = {"port": port_info.get("port")}
                if is_valid_key_in_dict(port_info, "protocol"):
                    _interface["protocol"] = port_info.get("protocol")

                if kwargs is not None:
                    _interface.update(kwargs)

                _interface_node = self.graph_db.get_node_args("i", "Interface", props=_interface)
                # MATCH (s:Service {...props})
                # CREATE (i:Interface {...props})<-[:EXPOSES]-(s)
                self.graph_db.run_statement(str(
                    _qb.match().node(**_args).create()
                    .node(**_interface_node).related_from("EXPOSES").node(ref_name="s")))

            if service.prebuilt:
                for _port in _data_interaction.ports:
                    _add_interface(_port)
            else:
                _passive_interactions = \
                    [p for p in _data_interaction.interactions if p.type == DataInteractionType.PASSIVE]
                for _port in _data_interaction.ports:
                    _port_matched = self._find_interaction_by_port(_passive_interactions, _port.get("port"))
                    if _port_matched is None:
                        # if port not found in interactions
                        _add_interface(_port)
                    else:
                        for p in _port_matched:
                            _kwargs = {
                                "protocol": p.interaction_type,
                                **p.interaction_details,
                            }
                            _add_interface(_port, _kwargs)

        # 2.2. Add actively accessed interfaces.
        # 3. Parse dependencies in deploy configs.
