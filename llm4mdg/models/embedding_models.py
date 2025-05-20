from enum import Enum

from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import OpenAIEmbeddings as EmbeddingOpenAI

from ..logs import logger


class OpenAIEmbeddingModel(Enum):
    """
    Enum class to represent OpenAI embedding models.
    """
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @classmethod
    def _missing_(cls, _):
        _default = OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL
        logger.warning(f"No model name specified, fallback to `{_default.value}`.")
        return _default


class OpenAIEmbedding(BaseModel):
    """
    Class for holding basic information about single OpenAI embedding model.
    Same structure as `openai_embedding` section in configuration file.
    """
    api_key: str = "sk-"
    base_url: str = "https://api.openai.com/v1"
    model: OpenAIEmbeddingModel = OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL
    m: EmbeddingOpenAI | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = EmbeddingOpenAI(
            api_key=self.api_key, base_url=self.base_url,
            model=self.model.value
        )
