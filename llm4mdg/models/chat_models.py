from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from ..logs import logger


class BaseChat(ABC, BaseModel):
    type: str

    @abstractmethod
    def instance(self) \
            -> Union[
                ChatOpenAI,
                ChatAnthropic,
                ChatGoogleGenerativeAI,
                ChatVertexAI
            ]:
        pass


class OpenAIChatModel(Enum):
    """
    Enum class to represent OpenAI chat models.
    """
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_0301 = "gpt-3.5-turbo-0301"
    GPT_35_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_35_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4 = "gpt-4"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_0314 = "gpt-4-0314"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_32K_0314 = "gpt-4-32k-0314"
    GPT_4_32K_0613 = "gpt-4-32k-0613"

    @classmethod
    def _missing_(cls, _):
        _default = OpenAIChatModel.GPT_4O_MINI
        logger.warning(f"No model name specified, fallback to `{_default.value}`.")
        return _default


class OpenAIChat(BaseChat):
    """
    Class for holding basic information about single OpenAI chat model.
    Same structure as `openai_chat` section in configuration file.
    """
    api_key: str = "sk-"
    base_url: str = "https://api.openai.com/v1"
    model: OpenAIChatModel = OpenAIChatModel.GPT_4O_MINI
    temperature: float = 0.0
    m: ChatOpenAI | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = ChatOpenAI(
            api_key=self.api_key, base_url=self.base_url,
            model=self.model.value, temperature=self.temperature
        )

    @property
    def instance(self):
        return self.m


class AnthropicChatModel(Enum):
    """
    Enum class to represent Anthropic chat models.
    """
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_SONNET_20240229 = "claude-3-sonnet-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"
    CLAUDE_2 = "claude-2"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_INSTANT_1 = "claude-instant-1"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"

    @classmethod
    def _missing_(cls, _):
        _default = AnthropicChatModel.CLAUDE_3_SONNET
        logger.warning(f"No model name specified, fallback to `{_default.value}`.")
        return _default


class AnthropicChat(BaseChat):
    api_key: str = "sk-"
    base_url: str = "https://api.anthropic.com/v1"
    model: AnthropicChatModel = AnthropicChatModel.CLAUDE_3_SONNET
    temperature: float = 0.0
    m: ChatAnthropic | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = ChatAnthropic(
            api_key=self.api_key, base_url=self.base_url,
            model=self.model.value, temperature=self.temperature
        )

    @property
    def instance(self):
        return self.m


class GoogleChatModel(Enum):
    """
    Enum class to represent Google chat models.
    """
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_1_5_PRO_001 = "gemini-1.5-pro-001"
    GEMINI_1_5_FLASH_001 = "gemini-1.5-flash-001"
    PALM_2_CHAT_BISON = "palm-2-chat-bison"
    PALM_2_CHAT_BISON_32K = "palm-2-chat-bison-32k"

    @classmethod
    def _missing_(cls, _):
        _default = GoogleChatModel.GEMINI_1_5_PRO_001
        logger.warning(f"No model name specified, fallback to `{_default.value}`.")
        return _default


class GoogleChat(BaseChat):
    api_key: str = "sk-"
    base_url: str = "https://generativelanguage.googleapis.com"
    model: GoogleChatModel = GoogleChatModel.GEMINI_1_5_PRO_001
    temperature: float = 0.0
    m: Union[ChatGoogleGenerativeAI, ChatVertexAI] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.model in [GoogleChatModel.PALM_2_CHAT_BISON, GoogleChatModel.PALM_2_CHAT_BISON_32K]:
            # Vertex Palm models
            self.m = ChatVertexAI(
                api_key=self.api_key, base_url=self.base_url,
                model=self.model.value, temperature=self.temperature
            )
        else:
            # Gemini models
            self.m = ChatGoogleGenerativeAI(
                api_key=self.api_key, base_url=self.base_url,
                model=self.model.value, temperature=self.temperature
            )

    @property
    def instance(self):
        return self.m
