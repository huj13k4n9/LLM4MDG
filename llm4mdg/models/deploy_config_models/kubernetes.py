import os
from typing import Dict, List, Any, Union

import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator

from .base import DeployConfig, DeployConfigType, PortMapping
from ...utils import is_valid_key_in_dict, is_valid_string


class KubernetesMetadata(BaseModel):
    """
    A representation class of a Kubernetes ObjectMeta.
    Reference: https://kubernetes.io/docs/reference/kubernetes-api/common-definitions/object-meta/
    """
    # For property `name`, should not be used directly
    object_name: str | None = Field(alias="name")
    generated_name: str | None = None

    namespace: str | None = None
    labels: Dict[str, str] | None = None

    @property
    def name(self):
        return (
            self.object_name
            if is_valid_string(self.object_name)
            else self.generated_name
        )

    @validator("namespace", pre=True, always=True)
    @classmethod
    def set_default_namespace(cls, v):
        if v is None or len(v) == 0:
            return "default"
        else:
            return v

    def name_matched(self, name: str):
        return (self.object_name == name or
                self.generated_name == name or
                (name in self.labels.values() if self.labels else False))


class KubernetesContainer(BaseModel):
    """
    A representation class of a Kubernetes Container.
    Reference: https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/pod-v1/#Container
    """
    name: str
    image: str | None = None
    ports: Union[List[Dict[str, Any]], List[PortMapping]] | None = None
    environment: List[Dict[str, Union[str, Any]]] | None = Field(alias="env")
    env_from: List[Dict[str, Union[str, Any]]] | None = Field(alias="envFrom")

    # Unimplemented: env.valueFrom, envFrom

    @validator("ports")
    @classmethod
    def transform_port_mapping(cls, v):
        if v is None or len(v) == 0:
            return None

        _ret: List[PortMapping] = []
        for port in v:
            _ret.append(PortMapping(
                host_port=port.get("hostPort") if is_valid_key_in_dict(port, "hostPort") else None,
                container_port=port.get("containerPort") if is_valid_key_in_dict(port, "containerPort") else None,
                protocol=port.get("protocol") if is_valid_key_in_dict(port, "protocol") else None,
            ))

        if len(_ret) == 0:
            return None
        return _ret


class KubernetesPod(BaseModel):
    """
    A representation class of a Kubernetes Pod.
    Reference: https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/pod-v1
    """
    metadata: KubernetesMetadata | None = None
    hostname: str | None = None
    subdomain: str | None = None
    aliases: List[Dict[str, Union[str, List[str]]]] | None = Field(alias="hostAliases")
    containers: List[KubernetesContainer] | None = None

    # Only used for parsing data
    spec: Dict[str, Any] | None = None

    @root_validator
    @classmethod
    def transform(cls, v):
        assert "spec" in v

        _spec = v.get("spec")
        v["hostname"] = _spec.get("hostname") if is_valid_key_in_dict(_spec, "hostname") else None
        v["subdomain"] = int(_spec.get("subdomain")) if is_valid_key_in_dict(_spec, "subdomain") else None
        v["aliases"] = _spec.get("aliases") if is_valid_key_in_dict(_spec, "aliases") else None
        v["containers"] = [KubernetesContainer(**v) for v in _spec.get("containers")] if is_valid_key_in_dict(_spec,
                                                                                                              "containers") else None
        return v


class KubernetesDeployment(BaseModel):
    """
    A representation class of a Kubernetes Deployment.
    Reference: https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/deployment-v1
    """
    metadata: KubernetesMetadata | None = None
    selector: Dict[str, str] | None = None
    replicas: int | None = None
    pod_template: KubernetesPod | None = None

    # Only used for parsing data
    spec: Dict[str, Any] | None = None

    @root_validator
    @classmethod
    def transform(cls, v):
        assert "spec" in v

        _spec = v.get("spec")
        v["selector"] = _spec.get("selector") if is_valid_key_in_dict(_spec, "selector") else None
        v["replicas"] = int(_spec.get("replicas")) if is_valid_key_in_dict(_spec, "replicas") else 1
        v["pod_template"] = KubernetesPod(**_spec.get("template")) if is_valid_key_in_dict(_spec, "template") else None
        return v


class KubernetesService(BaseModel):
    """
    A representation class of a Kubernetes Service.
    Reference: https://kubernetes.io/docs/reference/kubernetes-api/service-resources/service-v1
    """
    metadata: KubernetesMetadata | None = None
    selector: Dict[str, str] | None = None
    ports: Union[List[Dict[str, Any]], List[PortMapping]] | None = None
    type: str | None = None
    external_name: str | None = None

    # Only used for parsing data
    spec: Dict[str, Any] | None = None

    @root_validator
    @classmethod
    def transform(cls, v):
        assert "spec" in v

        _spec = v.get("spec")
        _service_type = _spec.get("type") if is_valid_key_in_dict(_spec, "type") else None

        v["type"] = _service_type
        v["selector"] = _spec.get("selector") if "selector" in _spec else None
        v["ports"]: List[PortMapping] = []

        if _service_type == "ExternalName" and is_valid_key_in_dict(_spec, "externalName"):
            v["external_name"] = _spec.get("externalName")
            v["ports"] = None
        elif is_valid_key_in_dict(_spec, "ports"):
            for port in _spec.get("ports"):
                _target_port = port.get("targetPort") \
                    if is_valid_key_in_dict(port, "targetPort") else None
                _port = port.get("port") \
                    if is_valid_key_in_dict(port, "port") else None
                _node_port = port.get("nodePort") \
                    if is_valid_key_in_dict(port, "nodePort") else None
                _protocol = port.get("protocol") \
                    if is_valid_key_in_dict(port, "protocol") else None
                _app_protocol = port.get("appProtocol") \
                    if is_valid_key_in_dict(port, "appProtocol") else None

                if _port:
                    try:
                        _port = int(_port)
                    except:
                        _port = None
                if _target_port:
                    try:
                        _target_port = int(_target_port)
                    except:
                        _target_port = None
                if _node_port:
                    try:
                        _node_port = int(_node_port)
                    except:
                        _node_port = None
                if _protocol or _app_protocol:
                    if _protocol:
                        _protocol += "" if not _app_protocol else _app_protocol
                    else:
                        _protocol = _app_protocol if _app_protocol else None

                if _target_port:
                    v["ports"].append(PortMapping(
                        host_port=_port,
                        container_port=_target_port,
                        protocol=_protocol))
                elif _port:
                    v["ports"].append(PortMapping(
                        host_port=_node_port,
                        container_port=_port,
                        protocol=_protocol))

        if len(v["ports"]) == 0:
            v["ports"] = None

        return v


class KubernetesDeployConfig(DeployConfig):
    k8s_pods: List[KubernetesPod] | None = None
    k8s_deployments: List[KubernetesDeployment] | None = None
    k8s_services: List[KubernetesService] | None = None

    @staticmethod
    def from_config(config_path: str) -> 'KubernetesDeployConfig':
        with open(config_path, "r") as f:
            config = [c for c in yaml.safe_load_all(f) if is_valid_key_in_dict(c, "kind")]

        _pods = [c for c in config
                 if c.get("kind") == "Pod" and "spec" in c and c.get("spec") is not None]
        _deployments = [c for c in config
                        if c.get("kind") == "Deployment" and "spec" in c and c.get("spec") is not None]
        _services = [c for c in config
                     if c.get("kind") == "Service" and "spec" in c and c.get("spec") is not None]

        if len(_pods) == 0 and len(_deployments) == 0 and len(_services) == 0:
            return None

        return KubernetesDeployConfig(
            path=os.path.abspath(config_path),
            type=DeployConfigType.KUBERNETES,
            k8s_pods=_pods,
            k8s_deployments=_deployments,
            k8s_services=_services,
        )
