import os
from typing import List

from langchain_core.pydantic_v1 import BaseModel

from .base import Action
from ..logs import logger, error_and_raise
from ..models.deploy_config_models import DockerComposeDeployConfig, \
    KubernetesDeployConfig, DeployConfigType, DeployConfig
from ..utils import absolute_path, relative_path


class ParseDeployConfigsAction(Action, BaseModel):
    project_loc: str
    deploy_configs: List[DeployConfig]

    def run(self) -> List[DeployConfig]:
        parsed_configs = []
        config_type_set = set([c.type for c in self.deploy_configs])

        # All configs are of unknown types
        if len(config_type_set) == 1 and DeployConfigType.UNKNOWN in config_type_set:
            logger.warning("Unknown type of config found, only {} are supported."
                           .format([c for c in DeployConfigType if c != DeployConfigType.UNKNOWN]))
            return parsed_configs

        for config in [c for c in self.deploy_configs if c.type != DeployConfigType.UNKNOWN]:
            def _process_config(c):
                if config.type == DeployConfigType.DOCKER_COMPOSE:
                    logger.info(f"Parsing docker compose deploy file: "
                                f"<yellow>{relative_path(self.project_loc, c)}</yellow>")
                    docker_compose_config = DockerComposeDeployConfig.from_config(c)
                    # Load info in `build` section if it exists.
                    docker_compose_config.load_build_context()

                    parsed_configs.append(docker_compose_config)
                elif config.type == DeployConfigType.KUBERNETES:
                    logger.info(f"Parsing kubernetes deploy file: "
                                f"<yellow>{relative_path(self.project_loc, c)}</yellow>")
                    parsed_configs.append(KubernetesDeployConfig.from_config(c))

            config_abs_path = absolute_path(self.project_loc, config.path)
            try:
                if os.path.isdir(config_abs_path):
                    for file in os.listdir(config_abs_path):
                        _process_config(absolute_path(config_abs_path, file))
                else:
                    _process_config(config_abs_path)
            except Exception as e:
                logger.warning("Error parsing config file/path {}: {}".format(
                    config_abs_path, str(e).replace('<', '\\<')
                ))
                continue

        return parsed_configs
