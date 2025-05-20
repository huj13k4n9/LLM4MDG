from abc import ABC, abstractmethod
from typing import List, ClassVar

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel


class Prompt(ABC):
    """
    A base class for handling different prompts.
    """

    @classmethod
    @abstractmethod
    def get_prompt(cls, **kwargs):
        pass


class BasicPrompt(Prompt, BaseModel):
    """
    A single piece of prompt template, no category specified.
    """
    prompt: ClassVar[PromptTemplate]

    @classmethod
    def get_prompt(cls, **kwargs) -> str:
        return cls.prompt.format(**kwargs)


class ChatPrompt(Prompt, BaseModel):
    """
    A single piece of a chat prompt template,
    containing a list of prompts, and can be in multi roles.
    """
    prompt: ClassVar[ChatPromptTemplate]

    @classmethod
    def get_prompt(cls, **kwargs) -> List[BaseMessage]:
        return cls.prompt.format_messages(**kwargs)
