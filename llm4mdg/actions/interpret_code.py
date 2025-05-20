import os
from typing import List, Tuple

from langchain_core.pydantic_v1 import BaseModel

from .base import Action
from .tools import call_llm_and_return_result
from ..logs import logger, error_and_raise
from ..models import BaseChat
from ..prompts import InterpretCodePrompt
from ..utils import relative_path


class InterpretCodeAction(Action, BaseModel):
    llm: BaseChat
    project_loc: str
    service_relative_path: str
    code_path: str
    dir_structure: str
    additional_configs: List[str] | None = None

    def run(self) -> Tuple[str, str | None]:
        if os.path.exists(self.code_path) and os.path.isfile(self.code_path):
            try:
                code_content = open(self.code_path, "r", encoding="utf-8").read()
            except UnicodeDecodeError:
                logger.warning(f"Possible binary file {self.code_path}, skipping.")
                return "", None

            if code_content is None or code_content == "":
                # If file is empty, skip interpreting and this file won't be saved in vector DB.
                return "", None

            rel_path = relative_path(self.project_loc, self.code_path)
            prompt = InterpretCodePrompt.get_prompt(
                dir_structure=self.dir_structure,
                relative_path=rel_path,
                code_content=code_content,
                additional_configs="\n".join(self.additional_configs) if self.additional_configs is not None else None,
            )
            _result = call_llm_and_return_result(self.llm, prompt)

            logger.info(f"Successfully got interpreted code of file <yellow>{rel_path}</yellow>")
            return code_content, _result
        else:
            logger.warning(f"Target code file {self.code_path} is not a file or it does not exist, skipping")
            return "", None
