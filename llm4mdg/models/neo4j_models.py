import re
from enum import Enum

from cymple import QueryBuilder
from langchain_core.pydantic_v1 import BaseModel, validator
from neo4j import GraphDatabase, kerberos_auth, bearer_auth, Driver, Session

from ..logs import error_and_raise, logger

# Reference: https://neo4j.com/docs/python-manual/current/connect-advanced/
# <SCHEME>://<HOST>[:<PORT>[?policy=<POLICY-NAME>]]
NEO4J_REGEX = (r"^((?:neo4j|bolt)(?:\+(?:s|ssc))?)://"  # Protocol
               r"([a-zA-Z0-9\.\-_]+|\[[0-9a-f:]+\])"  # Host
               r"(?::(\d{1,5}))?(?:\?policy=(.*?))?$")  # Port and Policy


class Neo4jAuthType(str, Enum):
    BASIC = "basic"
    KERBEROS = "kerberos"
    BEARER = "bearer"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return None


class Neo4jGraphDB(BaseModel):
    # Mandatory fields
    uri: str
    auth_type: Neo4jAuthType
    database: str | None = "neo4j"
    collection_name: str | None = None

    n: Driver | None = None
    session: Session | None = None
    qb: QueryBuilder = QueryBuilder()

    class Config:
        arbitrary_types_allowed = True

    # Values of fields below depend on `auth_type`
    username: str | None = None
    password: str | None = None
    kerberos_ticket: str | None = None
    bearer_token: str | None = None

    @staticmethod
    def get_node_args(ref_name, labels, props: dict = dict()):
        return {
            "labels": labels,
            "ref_name": ref_name,
            "properties": props,
        }

    @property
    def project_node(self):
        return self.get_node_args("p", "Project", {"identifier": self.collection_name})

    @validator("uri")
    @classmethod
    def validate_neo4j_uri(cls, value):
        assert re.fullmatch(NEO4J_REGEX, value) is not None
        return value

    def init_db(self):
        connection_args = {
            "connection_timeout": 5.0,
            "keep_alive": True,
        }

        if self.auth_type == Neo4jAuthType.BASIC:
            assert self.username is not None and self.password is not None
            self.n = GraphDatabase.driver(self.uri, auth=(self.username, self.password), **connection_args)
        elif self.auth_type == Neo4jAuthType.KERBEROS:
            assert self.kerberos_ticket is not None
            self.n = GraphDatabase.driver(self.uri, auth=kerberos_auth(self.kerberos_ticket), **connection_args)
        elif self.auth_type == Neo4jAuthType.BEARER:
            assert self.bearer_token is not None
            self.n = GraphDatabase.driver(self.uri, auth=bearer_auth(self.bearer_token), **connection_args)

        try:
            self.n.verify_connectivity()
        except Exception as e:
            error_and_raise(f"Error connecting to Neo4j: {e}")

        self.session = self.n.session(database=self.database)

    def close_db(self):
        self.session.close()
        self.n.close()

    def get_data_and_count(self, query: str):
        _escaped = query.replace('<', '\\<')
        logger.debug(f"Cypher to execute: `{_escaped}`")
        _ret = self.session.run(query)
        _ret = [_r for _r in _ret]
        return _ret, len(_ret)

    def run_statement(self, statement: str):
        _escaped = statement.replace('<', '\\<')
        logger.debug(f"Cypher to execute: `{_escaped}`")
        self.session.run(statement)

    def init_collection(self):
        _, count = self.get_data_and_count(
            str(self.qb.match().node(**self.project_node).return_literal("p")))

        if count == 0:
            logger.info(f"Project node {self.collection_name} not found, creating one.")
            # Create a `Project` node as identifier of this project
            self.run_statement(str(self.qb.create().node(**self.project_node)))
        else:
            logger.info(f"Project node {self.collection_name} already exists, using existing one.")

    def reset_collection(self):
        # Delete Interface nodes having relationship with all Service nodes.
        # MATCH (s:Service)-[r]->(i:Interface) DELETE r,i
        self.run_statement(str(self.qb.match().node(**self.get_node_args("s", "Service"))
                               .related_to(ref_name="r").node(ref_name="i", labels="Interface").delete("r, i")))

        # Delete Service nodes having relationship with Project node.
        # MATCH (s:Service)->[r]->(:Project {...props}) DELETE s, r
        self.run_statement(str(self.qb.match().node(**self.get_node_args("s", "Service"))
                               .related_to(ref_name="r").node(**self.project_node).delete("s, r")))

        # Then delete the Project node.
        # MATCH (p:Project {...props}) DELETE p
        self.run_statement(str(self.qb.match().node(**self.project_node).delete(ref_name="p")))
        logger.info(f"Data of Project {self.collection_name} reset successfully.")
