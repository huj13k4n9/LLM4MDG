import os
from typing import Dict, List

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..logs import error_and_raise
from ..models import BaseChat


class ReturnResultTool(BaseModel):
    """Return the result of task in JSON format, use argument `result` to store the result."""
    result: str = Field(description="The result of task in JSON format")


class ListDirectoryTool(BaseModel):
    """List the items of given directory using command `tree`."""
    path: str = Field(description="The directory to list")

    def call(self):
        try:
            return os.popen(f"tree {self.path}").read()
        except Exception as e:
            return "Error: " + str(e)


class ReadFileTool(BaseModel):
    """Get the content of a given file."""
    path: str = Field(description="The file to read")

    def call(self):
        try:
            return open(self.path, "r").read()
        except Exception as e:
            return "Error: " + str(e)


class GetCodeTool(BaseModel):
    """Get the content of a given code/configuration path in vector DB."""
    path: str = Field(description="The file path to get content")


class ToolCall(BaseModel):
    args: Dict[str, str]
    id: str
    type: str


class ToolCallList(BaseModel):
    tool_calls: List[ToolCall]


def call_llm_and_return_result(llm: BaseChat, prompt: List[BaseMessage], result_class=None):
    llm_with_tools = llm.instance.bind_tools([ReturnResultTool], tool_choice="ReturnResultTool")
    parser = PydanticToolsParser(tools=[ReturnResultTool])

    ai_msg = llm_with_tools.invoke(prompt)
    if not hasattr(ai_msg, "tool_calls"):
        error_and_raise("No tools called")

    ret = parser.invoke(ai_msg)
    if len(ret) != 1 or not isinstance(ret[0], ReturnResultTool):
        error_and_raise("Unexpected tool call returned")

    return result_class.parse_raw(ret[0].result) if result_class is not None else ret[0].result
