import os
import re
from io import StringIO
from typing import Union, List, Dict, Any, Tuple

import yaml
from dockerfile_parse import DockerfileParser
from dotenv import dotenv_values
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator, Field

from ...utils import absolute_path
from .base import PortMapping, HOSTS_REGEX, PORT_MAPPING_REGEX, PORT_EXPOSE_REGEX, \
    DOCKERFILE_FROM_REGEX, DeployConfig, DeployConfigType
from ...logs import logger


class DockerComposeDeployment(BaseModel):
    name: str
    config_loc: Union[str, List[str]] | None = None
    aliases: List[str] | None = None
    build: Union[str, Dict[str, Any]] | None = None
    environment: Union[List[str], Dict[str, Union[str, List[str], None]]] | None = None
    extends: Dict[str, str] | None = None  # Implement this later
    extra_hosts: Union[List[str], Dict[str, Union[str, List[str]]]] | None = None
    image: Union[str, List[str]] | None = None
    ipc: Union[str, List[str]] | None = None
    networks: Union[List[str], Dict[str, Any]] | None = None
    ports: Union[List[str], List[PortMapping]] | None = None
    depends_on: Union[List[str], Dict[str, Any]] | None = None

    @root_validator(pre=True)
    @classmethod
    def transform_attributes(cls, value):
        """
        Process the following attributes:
        aliases, expose, hostname, extra_hosts, networks, environment, env_file
        """
        assert "config_loc" in value
        config_dir = value.get("config_loc")

        def _get_attr(name: str):
            return value.get(name) if name in value else None

        def _process_envs(environment, env_file) -> Dict[str, str] | None:
            _environment: Dict[str, str] = {}
            env_files: List[str] = []

            if environment is not None and isinstance(environment, list):
                all_env_str = "\n".join(environment)
                env_dict = dotenv_values(stream=StringIO(all_env_str), interpolate=False)
                _environment = dict(env_dict)
            elif environment is not None and isinstance(environment, dict):
                _environment = environment

            if env_file is not None and isinstance(env_file, str):
                env_files.append(absolute_path(config_dir, env_file))
            elif env_file is not None and isinstance(env_file, list):
                for e in env_file:
                    if isinstance(e, str):
                        env_files.append(absolute_path(config_dir, e))
                    if isinstance(e, dict):
                        assert "path" in e and isinstance(e.get("path"), str)
                        env_files.append(absolute_path(config_dir, e.get("path")))

            # Load each env_file
            for file in env_files:
                envs = dotenv_values(file, interpolate=False)
                # `environment` has higher priority than `env_file`
                envs.update(_environment)
                _environment = envs

            return dict(_environment) if len(_environment) > 0 else None

        def _process_expose(expose, ports) -> List[str] | None:
            _ports: List[str] = []
            if ports is not None and isinstance(ports, list):
                _ports = ports

            if expose is not None and isinstance(expose, list):
                for e in expose:
                    _ports.append(str(e))

            return _ports if len(_ports) > 0 else None

        def _process_extra_hosts(extra_hosts) -> Dict[str, str] | None:
            _extra_hosts: Dict[str, str] = {}

            if extra_hosts is not None and isinstance(extra_hosts, list):
                for host in extra_hosts:
                    if isinstance(host, str):
                        matches = re.finditer(HOSTS_REGEX, host)
                        for match in matches:
                            # Find custom host resolution
                            hostname, addr = match.groups()
                            _extra_hosts[hostname] = addr

            return _extra_hosts if len(_extra_hosts) > 0 else None

        def _process_networks_and_aliases(networks, hostname) -> Tuple[List[str], List[str]] | None:
            _aliases: List[str] = []
            _networks: List[str] = []

            if hostname is not None and isinstance(hostname, str):
                _aliases.append(hostname)

            if networks is not None and isinstance(networks, dict):
                for network in networks.items():
                    network_name, network_data = network
                    # Save network names of current service
                    _networks.append(network_name)

                    # Save service aliases on specific networks
                    if "aliases" in network_data and isinstance(network_data.get("aliases"), list):
                        for a in network_data.get("aliases"):
                            if isinstance(a, str):
                                _aliases.append(a)

            return _aliases if len(_aliases) > 0 else None, _networks if len(_networks) > 0 else None

        value["environment"] = _process_envs(
            environment=_get_attr("environment"), env_file=_get_attr("env_file"))
        value["ports"] = _process_expose(
            expose=_get_attr("expose"), ports=_get_attr("ports"))
        value["extra_hosts"] = _process_extra_hosts(extra_hosts=_get_attr("extra_hosts"))
        value["aliases"], value["networks"] = _process_networks_and_aliases(
            networks=_get_attr("networks"), hostname=_get_attr("hostname"))

        return value

    @validator("build", pre=True, always=True)
    @classmethod
    def validate_build(cls, v):
        if v is not None:
            if isinstance(v, str) and v == "":
                return None
            if isinstance(v, dict):
                assert "context" in v
                if v.get("context") is None or v.get("context") == "":
                    return None
        return v

    @validator("ports")
    @classmethod
    def transform_port_mapping(cls, v):
        if isinstance(v, list) and all(isinstance(item, PortMapping) for item in v):
            return v

        if v is None or (v is not None and isinstance(v, list) and len(v) == 0):
            return None

        assert isinstance(v, list) and all(isinstance(item, str) for item in v)

        ret: List[PortMapping] = []
        for item in v:
            matches = re.finditer(PORT_MAPPING_REGEX, item)
            for match in matches:
                host_port, host_port_end, container_port, container_port_end, protocol = match.groups()

                # Only container port, no range
                if (
                        host_port is not None and
                        host_port_end is None and
                        container_port is None and
                        container_port_end is None
                ):
                    ret.append(PortMapping(
                        container_port=int(host_port), protocol=protocol))
                # Only container port range
                elif (
                        host_port is not None and
                        host_port_end is not None and
                        container_port is None and
                        container_port_end is None
                ):
                    for i in range(int(host_port), int(host_port_end) + 1):
                        ret.append(PortMapping(
                            container_port=i, protocol=protocol))
                # Container port and host port mapping
                elif (
                        host_port is not None and
                        host_port_end is None and
                        container_port is not None and
                        container_port_end is None
                ):
                    ret.append(PortMapping(
                        host_port=int(host_port), container_port=int(container_port), protocol=protocol))
                # Container port and host port range mapping
                elif (
                        host_port is not None and
                        host_port_end is not None and
                        container_port is not None and
                        container_port_end is not None
                ):
                    host_port_num, host_port_end_num, container_port_num, container_port_end_num = \
                        int(host_port), int(host_port_end), int(container_port), int(container_port_end)
                    assert host_port_end_num - host_port_num == container_port_end_num - container_port_num

                    while host_port_num <= host_port_end_num and container_port_num <= container_port_end_num:
                        ret.append(PortMapping(
                            host_port=host_port_num, container_port=container_port_num, protocol=protocol))
                        host_port_num += 1
                        container_port_num += 1
                # Bind multi host ports on the same container port
                elif (
                        host_port is not None and
                        host_port_end is not None and
                        container_port is not None and
                        container_port_end is None
                ):
                    for i in range(int(host_port), int(host_port_end) + 1):
                        ret.append(PortMapping(
                            host_port=i, container_port=int(container_port), protocol=protocol))

        assert len(ret) != 0
        return ret

    @validator("depends_on", pre=True)
    @classmethod
    def get_depends_on(cls, v):
        if v is not None:
            if isinstance(v, list):
                return v
            elif isinstance(v, dict):
                _ret: List[str] = []
                for k, _ in v.items():
                    _ret.append(k)
                return _ret
        return v

    def load_build_context(self):
        """
        Load info in Dockerfile specified in `build` section of current deployment.
        This method should not be called in a merged deployment.
        """

        def _parse_dockerfile(path: str):
            # Parse Dockerfile, extract EXPOSE, ENV, ARG, FROM
            logger.debug(f"Parsing dockerfile {path}")
            with open(path, "rb") as dockerfile:
                parser = DockerfileParser(fileobj=dockerfile)

                _args = [i.get("value") for i in parser.structure if i.get("instruction") == "ARG"]
                _expose = [i.get("value") for i in parser.structure if i.get("instruction") == "EXPOSE"]
                _envs = parser.envs
                _images = [i.get("value") for i in parser.structure if i.get("instruction") == "FROM"]
                _ports: List[PortMapping] = []

                # Load arguments as dict
                _arg_dict = dotenv_values(stream=StringIO("\n".join(_args)), interpolate=False)

                # Transform `expose` value to PortMapping
                for port in _expose:
                    _p = port.split(" ")
                    for pp in _p:
                        container_port, container_port_end, protocol = re.match(PORT_EXPOSE_REGEX, pp).groups()
                        if container_port_end is not None:
                            assert int(container_port_end) >= int(container_port)
                            for i in range(int(container_port), int(container_port_end) + 1):
                                _ports.append(PortMapping(container_port=i, host_port=None, protocol=protocol))
                        else:
                            _ports.append(PortMapping(container_port=int(container_port), host_port=None, protocol=protocol))

                # Update environment variables
                if len(_envs) != 0:
                    if self.environment is None:
                        self.environment = _envs
                    else:
                        self.environment.update(_envs)

                # Process `FROM xxx AS xxx`
                _from, _as = [], []
                _images_without_as = []
                for i in _images:
                    matches = re.match(DOCKERFILE_FROM_REGEX, i.strip())
                    if matches:
                        matches = matches.groups()

                        # Replace variables in Dockerfile
                        if matches[1] is not None:
                            _from.append(re.sub(r'\$\{(\S+)\}', lambda match: _arg_dict.get(match.group(1)), matches[0]))
                            _as.append(re.sub(r'\$\{(\S+)\}', lambda match: _arg_dict.get(match.group(1)), matches[1]))
                        else:
                            _images_without_as.append(matches[0])

                # Remove useless aliases defined by `AS`
                for i in range(len(_from) - 1, -1, -1):
                    if _from[i] in _as:
                        pass
                    else:
                        _images_without_as.append(_from[i])

                # Update image information
                if len(_images_without_as) != 0:
                    if self.image is None:
                        self.image = _images_without_as[0] if len(_images_without_as) == 1 else _images_without_as
                    elif isinstance(self.image, str):
                        _images_without_as.append(self.image)
                        self.image = _images_without_as

                # Update ports
                if len(_ports) != 0:
                    if self.ports is None:
                        self.ports = _ports
                    else:
                        self.ports.extend(_ports)

        # No `build` section
        if self.build is None:
            return

        if isinstance(self.build, str):
            context = absolute_path(self.config_loc, self.build)
            dockerfile_loc = absolute_path(context, "./Dockerfile")
            _parse_dockerfile(dockerfile_loc)
        elif isinstance(self.build, dict):
            context = absolute_path(self.config_loc, self.build.get("context"))
            if "dockerfile" in self.build:
                dockerfile_loc = absolute_path(context, self.build.get("dockerfile"))
                _parse_dockerfile(dockerfile_loc)
            else:
                dockerfile_loc = absolute_path(context, "./Dockerfile")
                _parse_dockerfile(dockerfile_loc)


class DockerComposeDeployConfig(DeployConfig):
    """
    A representation of a single docker-compose configuration file.
    Only `services` section and `networks` section are used, others are ignored.
    Reference: https://github.com/compose-spec/compose-spec
    """

    docker_deployments: Union[List[DockerComposeDeployment], Dict[str, Any]] | None = Field(alias="services")
    docker_networks: Union[List[str], Dict[str, Any]] | None = Field(alias="networks")

    @staticmethod
    def from_config(config_path: str) -> 'DockerComposeDeployConfig':
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return DockerComposeDeployConfig(
            path=os.path.abspath(config_path), type=DeployConfigType.DOCKER_COMPOSE, **config)

    @validator("docker_networks", pre=True, always=True)
    @classmethod
    def get_network_names(cls, v):
        if v is None or isinstance(v, list):
            return v

        ret: List[str] = []
        for network in v.keys():
            ret.append(network)
        return ret

    @validator("docker_deployments", pre=True, always=True)
    @classmethod
    def transform_dict_to_deployment(cls, v, values):
        if v is None or isinstance(v, list):
            return v

        assert "path" in values
        config_loc = os.path.dirname(values.get("path"))

        ret: List[DockerComposeDeployment] = []
        for d in v.items():
            name, data = d
            ret.append(DockerComposeDeployment(name=name, config_loc=config_loc, **data))
        return ret

    def load_build_context(self):
        """
        Load info in Dockerfile specified in `build` section of each deployment.
        This method should not be called in a merged config.
        """
        for d in self.docker_deployments:
            d.load_build_context()
