from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate

from .base import ChatPrompt

_prompt = """You are an expert in the field of computer science, currently responsible for code analysis of microservice projects. Based on the code given below, please follow the instructions and provide accurate and realistic results.

TASK INSTRUCTION: You need to analyze the following code or configuration file enclosed by triple backticks, which contains the content of a complete file from the project. Please analyze the content and derive the following results:
1. Explain the definition and functionality of this file based on the content, including but not limited to functions, classes, dependencies, etc.
2. Identify whether the code in the file expose interfaces to external services, including but not limited to HTTP APIs, gRPC interfaces, controllers, routes, listening ports, etc. If found, list relevant details like URI, host, port, request method, etc.
3. Identify whether the code in the file proactively interact with any external services, including but not limited to consuming HTTP APIs or gRPC interfaces, connecting to specific external ports, invoking SOAP services, using message queues, etc. If found, list relevant details like URI, host, port, request method, etc.
4. If possible, identify the framework, other open-source common services, programming language used by this project, based on the file content.
5. Please ensure your response is in natural language format, make it concise, precise, and clear, containing only relevant information in order to save tokens.
"""


class InterpretCodePrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _prompt),
        ("human", """# PROJECT DIRECTORY STRUCTURE:
```
{{ dir_structure }}

## ADDITIONAL CONFIGS
{{ additional_configs }}
```

# CODE CONTENT: (RELATIVE PATH:  `{{ relative_path }}`)
```
{{ code_content }}
```

ANSWER:""")], template_format="jinja2")
