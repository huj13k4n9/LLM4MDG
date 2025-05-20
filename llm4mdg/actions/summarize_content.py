from langchain_core.pydantic_v1 import BaseModel

from .base import Action
from .tools import call_llm_and_return_result
from ..logs import logger
from ..models import BaseChat
from ..prompts import SummarizeContentPrompt


class SummarizeContentAction(Action, BaseModel):
    llm: BaseChat
    brief: str
    content: str
    key_topics: str

    def run(self):
        prompt = SummarizeContentPrompt.get_prompt(
            brief=self.brief, content=self.content, key_topics=self.key_topics)
        _ret = call_llm_and_return_result(self.llm, prompt)
        logger.info("Messages summarized completely.")
        return _ret
