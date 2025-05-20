from enum import Enum
from typing import Dict, Any, List

from langchain_core.pydantic_v1 import BaseModel


class DataInteractionType(Enum):
    PASSIVE = "passive"
    ACTIVE = "active"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return None


class DataInteractionDirection(Enum):
    REQUEST_RESPONSE = "request-response"
    ONLY_SEND = "only-send"
    ONLY_RECEIVE = "only-receive"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return None


class DataInteraction(BaseModel):
    type: DataInteractionType | None = None
    directionality: DataInteractionDirection | None = None
    description: str | None = None
    target_service: str | None = None
    interaction_type: str | None = None
    interaction_details: Dict[str, Any] | None = None

    def __str__(self):
        return (f"Type: {self.type.value}, "
                f"Data Direction: {self.directionality.value}, "
                f"Target Service: {self.target_service}, "
                f"Interaction Type: {self.interaction_type}, "
                f"Details: {self.interaction_details}")


class ServiceAnalysis(BaseModel):
    # This value will be filled after the object is created, but its value will never be None.
    service_name: str | None = None

    analysis: str
    service: str | None = None
    type: str | None = None
    ports: List[Dict[str, Any]] | None = None


class PrebuiltServiceAnalysis(ServiceAnalysis):
    pass


class NonPrebuiltServiceAnalysis(ServiceAnalysis):
    interactions: List[DataInteraction] | None = None
    language: List[str] | None = None
