from typing import Dict, List

from langchain_core.pydantic_v1 import BaseModel, validator

from .deploy_config_models import DeployConfig


class IdentifiedService(BaseModel):
    name: str
    prebuilt: bool
    evidence: str
    source_dir: str | None = None
    configs: List[str] | None = None


class IdentifyServiceResult(BaseModel):
    deploy_config: List[DeployConfig]
    services: List[IdentifiedService]


class ValidatedResult(BaseModel):
    modification: str
    validated_result: IdentifyServiceResult


class ProcessConfigCenterResult(BaseModel):
    store: str
    analysis: str
    services_with_configs: Dict[str, List[str]] | None = None

    @validator("store")
    @classmethod
    def validate_store_method(cls, value: str) -> str:
        assert value.upper() == "LOCAL" or value.upper() == "REMOTE"
        return value
