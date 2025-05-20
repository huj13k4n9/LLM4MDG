from abc import ABC, abstractmethod
from typing import Dict, Union, List
from uuid import uuid4

import chromadb
import chromadb.config
from langchain_chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel, validator, root_validator
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings as EmbeddingOpenAI

from ..constant import INTERMEDIATE_DATA_LOC, VECTORDB_PRIMARY_FIELD, VECTORDB_TEXT_FIELD, VECTORDB_VECTOR_FIELD
from ..logs import error_and_raise


class BaseVecDB(ABC, BaseModel):
    db_type: str
    connection_type: str
    collection_name: str | None = None
    embd_func: EmbeddingOpenAI | None = None

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def init_db(self):
        pass

    @abstractmethod
    def add_data(self, datas, metadatas, ids=None) -> List[str]:
        pass

    @abstractmethod
    def delete_data(self, ids: List[str]):
        pass

    @abstractmethod
    def get_data_count(self, filter_condition: Dict[str, str]) -> int:
        pass

    @abstractmethod
    def retrieve_data(
            self,
            query: str,
            top_k: int,
            filter_condition: Dict[str, str],
            search_type: str
    ):
        pass


class ChromaVecDB(BaseVecDB):
    connection_config: Dict[str, Union[str, int, bool]] | None = None
    v: Chroma | None = None

    @root_validator
    @classmethod
    def validate(cls, value):
        # Check connection_type first
        connection_type = value.get("connection_type").lower()
        assert connection_type in ["in-memory", "local", "remote"], "Unexpected connection type."

        if connection_type == "remote":
            config = value.get("connection_config")
            assert "host" in config and config.get("host") != "", "Remote connection `host` is missing."
            assert "port" in config and config.get("port") != "", "Remote connection `port` is missing."
            assert "ssl" in config and config.get("ssl") is bool, "Remote connection `ssl` is missing."

        return value

    def init_db(self):
        client = None
        config = self.connection_config

        if self.connection_type == "in-memory":
            client = chromadb.EphemeralClient(settings=chromadb.config.Settings())
        elif self.connection_type == "local":
            if config is not None and "path" in config and config.get("path") != "":
                path = config.get("path")
            else:
                path = str(INTERMEDIATE_DATA_LOC / f"{self.collection_name}.db")
            client = chromadb.PersistentClient(path=path, settings=chromadb.config.Settings())
        elif self.connection_type == "remote":
            host, port, ssl = config.get("host"), config.get("port"), config.get("ssl")
            client = chromadb.HttpClient(
                host=host, port=port, ssl=ssl, settings=chromadb.config.Settings(allow_reset=True))

        if client is None:
            error_and_raise("Chroma client initialization failed.")
        else:
            self.v = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embd_func,
                client=client,
            )

    def add_data(self, datas, metadatas, ids=None) -> List[str]:
        if len(datas) != len(metadatas):
            error_and_raise("Unmatched length of input data and metadata")

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(datas))]
        return self.v.add_texts(texts=datas, metadatas=metadatas, ids=ids)


class MilvusVecDB(BaseVecDB):
    connection_uri: str | None = None
    v: Milvus | None = None

    @validator('connection_type', pre=True, always=True)
    @classmethod
    def check_connection_type(cls, value: str) -> str:
        assert value.lower() in ["local", "remote"], "Unexpected connection type."
        return value

    @staticmethod
    def _expr(expr_dict: Dict[str, str]) -> str:
        expr = []
        for c in expr_dict.items():
            key, value = c
            expr.append(f'{key} == "{value}"')
        return " && ".join(expr)

    def init_db(self):
        default_params = {
            "embedding_function": self.embd_func,
            "collection_name": self.collection_name,
            "primary_field": VECTORDB_PRIMARY_FIELD,
            "text_field": VECTORDB_TEXT_FIELD,
            "vector_field": VECTORDB_VECTOR_FIELD,
        }

        if self.connection_uri is not None and self.connection_uri != "":
            self.v = Milvus(
                connection_args={"uri": self.connection_uri},
                **default_params,
            )
        else:
            self.v = Milvus(
                connection_args={"uri": str(INTERMEDIATE_DATA_LOC / f"{self.collection_name}.db")},
                **default_params,
            )

    def add_data(self, datas, metadatas, ids=None) -> List[str]:
        if len(datas) != len(metadatas):
            error_and_raise("Unmatched length of input data and metadata")

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(datas))]
        return self.v.add_texts(texts=datas, metadatas=metadatas, ids=ids)

    def delete_data(self, ids: List[str]):
        return self.v.delete(ids)

    def get_data_count(self, filter_condition: Dict[str, str]) -> int:
        expr = []
        for c in filter_condition.items():
            key, value = c
            expr.append(f'{key} == "{value}"')

        ret = self.v.get_pks(self._expr(filter_condition))
        return len(ret) if ret is not None else 0

    def retrieve_data(
            self,
            query: str,
            top_k: int,
            filter_condition: Dict[str, str],
            search_type: str = "mmr"
    ):
        return self.v.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": top_k,
                "expr": self._expr(filter_condition),
            }
        ).invoke(query)
