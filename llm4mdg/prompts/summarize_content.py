from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate

from .base import ChatPrompt

_prompt = """# Task Instructions
You are an AI assistant specializing in the field of computer science. You will receive a text related to the computer science domain along with a brief introduction to its content. Due to the extensive nature of the content, it exceeds the maximum context length that large language models can process in a single interaction. Your task is to distill and condense this text, retaining only the key information (which is shown below) while removing irrelevant and redundant details. Specific requirements are as follows:

- When condensing the text, ensure the accuracy of the original statements. There should be no omissions or errors.
- Unless data format is specified, reply with natural language. If given text is segmented by multiple blocks, return text blocks with the same format.
- Please note that content related to the topics listed below must be retained, DON'T LOST ANY DETAILS ABOUT THEM !!! Failure to do so will result in your dismissal.
```
{{ key_topics }}
```
"""


class SummarizeContentPrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _prompt),
        ("human", """# Brief Description on Given Text
{{ brief }}

# Text Content
{{ content }}""")], template_format="jinja2")
