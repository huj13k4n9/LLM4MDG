import os

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.pydantic_v1 import BaseModel

from .base import Action
from .summarize_content import SummarizeContentAction
from .tools import ReturnResultTool, ListDirectoryTool, ReadFileTool, ToolCallList
from ..constant import AGENT_MAX_ITER_TIMES
from ..logs import logger, error_and_raise
from ..models import ProcessConfigCenterResult, IdentifyServiceResult, BaseChat
from ..prompts import ProcessConfigCenterPrompt
from ..utils import absolute_path, is_valid_string

_tools = [ListDirectoryTool, ReadFileTool, ReturnResultTool]
_tools_str = {
    "ListDirectoryTool": ListDirectoryTool,
    "ReadFileTool": ReadFileTool,
    "ReturnResultTool": ReturnResultTool,
}


def summarize_llm_messages(llm, msgs):
    summarized_content = SummarizeContentAction(
        llm=llm,
        brief="This text includes a list of chat history messages from a human and a LLM.",
        content=msgs,
        key_topics="\n".join([
            "Summarize the whole process of chat history, "
            "make sure the behaviour that each message has done is fully, precisely and clearly stated.",
        ]),
    ).run()
    return summarized_content


class ProcessConfigCenterAction(Action, BaseModel):
    iter_times: int = AGENT_MAX_ITER_TIMES
    llm: BaseChat
    identified_result: IdentifyServiceResult
    config_center_name: str = None
    config_center_dir: str | None = None
    project_loc: str

    def _format_services(self) -> str:
        ret = ""
        for service in self.identified_result.services:
            ret += f"- {service.name}"
            if not service.prebuilt:
                ret += f" (source: {service.source_dir})"
            if service.name == self.config_center_name:
                ret += " [CONFIG_CENTER]"
            ret += "\n"
        return ret

    def run(self) -> ProcessConfigCenterResult:
        if not is_valid_string(self.config_center_name):
            error_and_raise("Config center name is unknown")

        source_dir = ""
        if is_valid_string(self.config_center_dir):
            source_dir = absolute_path(self.project_loc, self.config_center_dir)
        else:
            source_dir = ""
            for service in self.identified_result.services:
                if self.config_center_name == service.name:
                    source_dir = service.source_dir
                    break
        if not is_valid_string(source_dir):
            error_and_raise("Config center directory not found")

        cwd = os.getcwd()
        llm_with_tools = self.llm.instance.bind_tools(_tools)
        config_center_abs_path = absolute_path(self.project_loc, source_dir)
        os.chdir(config_center_abs_path)
        dir_structure = os.popen(f"tree {config_center_abs_path}").read()
        messages = ProcessConfigCenterPrompt.get_prompt(
            dir_structure=dir_structure, services=self._format_services())
        parser = JsonOutputToolsParser(return_id=True)
        system_message_length = len(messages)

        for i in range(self.iter_times):
            logger.info(f"ProcessConfigCenterAgent iteration round {i + 1} (max {self.iter_times})")
            ai_msg = llm_with_tools.invoke(messages)

            if not hasattr(ai_msg, "tool_calls"):
                error_and_raise("No tools called")

            if len(messages) > (25 + system_message_length):
                # Summarize first 10 items of chat history
                logger.info("Messages too long, summarize the first 10 messages.")
                initial_messages = messages[:system_message_length]
                messages_to_summarize = messages[system_message_length:system_message_length + 10]
                other_messages = messages[system_message_length + 10:]

                summarized = summarize_llm_messages(self.llm, str(messages_to_summarize))
                initial_messages.append(HumanMessage(content="# Previous chat history\n\n" + summarized))
                initial_messages.extend(other_messages)

                messages = initial_messages
                system_message_length += 1

            messages.append(ai_msg)
            ret = parser.invoke(ai_msg)
            tool_call_list = ToolCallList(tool_calls=ret)

            for tool_call in tool_call_list.tool_calls:
                selected_tool = _tools_str[tool_call.type]
                if selected_tool is ReturnResultTool:
                    logger.info(f"ProcessConfigCenterAgent returned result, used {i + 1} rounds.")
                    os.chdir(cwd)
                    return ProcessConfigCenterResult.parse_raw(selected_tool(**tool_call.args).result)
                else:
                    logger.info(f"ProcessConfigCenterAgent called {tool_call.type}")
                    logger.debug(f"ProcessConfigCenterAgent called {tool_call.type}({tool_call.args})")
                    tool_output = selected_tool(**tool_call.args).call()
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call.id))

        error_and_raise("Agent iteration completed, but no result returned.")