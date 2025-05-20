from enum import Enum
from typing import List

from langchain_core.pydantic_v1 import BaseModel, validator

PORT_MAPPING_REGEX = (r"^(?:\d{1,3}(?:\.\d{1,3}){3}:)?"  # Host IP (ignored)
                      r"(?:(\d{1,5})(?:-(\d{1,5}))?)"  # Host port range
                      r"(?::(?:(\d{1,5})(?:-(\d{1,5}))?)"  # Container port range
                      r"(?:/(tcp|udp))?)?$")  # Protocol
PORT_EXPOSE_REGEX = r"^(?:(\d{1,5})(?:-(\d{1,5}))?)(?:/(tcp|udp))?$"
HOSTS_REGEX = r"^([\w.-]+)(?:=|:)((?:\d{1,3}(?:\.\d{1,3}){3})|\[?(?:[a-fA-F0-9:]+)\]?)$"
DOCKERFILE_FROM_REGEX = r"^(\S+)(?: (?:AS|as|As|aS) (\S+))?$"


class DeployConfigType(str, Enum):
    DOCKER_COMPOSE = "docker"
    KUBERNETES = "kubernetes"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return None


class DeployConfig(BaseModel):
    path: str | List[str]
    type: DeployConfigType


class PortMapping(BaseModel):
    host_port: int | None = None
    container_port: int
    protocol: str | None = None

    def __eq__(self, other):
        if not isinstance(other, PortMapping):
            return NotImplemented

        return (self.host_port == other.host_port and
                self.container_port == other.container_port and
                self.protocol == other.protocol)

    def __hash__(self):
        return hash((self.host_port, self.container_port, self.protocol))

    def __str__(self):
        _ret = str(self.container_port)
        if self.host_port is not None:
            _ret = str(self.host_port) + ":" + _ret
        if self.protocol is not None:
            _ret += f"/{self.protocol}"
        return _ret

    @validator("protocol", pre=True, always=True)
    @classmethod
    def check_protocol(cls, v):
        if v is not None:
            return v.upper()
        return v

    @validator("container_port", "host_port", pre=True)
    @classmethod
    def check_port(cls, v):
        if v is not None:
            assert 1 <= v <= 65535
        return v
