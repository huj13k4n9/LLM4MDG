from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate

from .base import ChatPrompt

_prompt = """# Task Instructions
You are an AI assistant for microservices project analysis. You have data on all service instances in an open-source microservice-based project, including a configuration center that centralizes configuration data storage. However this centralization causes a lack of specific configuration details for individual services. Thus, please analyze the directory structure and file contents in the configuration center source directory to:

- Please analyze only the configuration files intended for the configuration center. Avoid analyzing configuration files for other services that may be in the same directory, to prevent potential misguidance.
- Determine how the configuration center stores configuration files: locally (Local) or remotely (Remote) in locations like Git repositories or databases.
- If the configuration center stores configurations locally, associate each configuration file with the corresponding service using it, based on the provided information. Note that there may be common configuration files used by all services, which should be associated with every service.
- Once you found association between configuration files and services, change the path of configuration files to relative path starting with `./` based on the root directory of configuration center.
- Please minimize the number of tool calls as much as possible to save tokens.

# Output Formats
Your output should strictly obey the following JSON format. Call `ReturnResultTool` to return your result.
```
{
    "store": "The way the configuration center stores configuration data, only two values available: `LOCAL` and `REMOTE`",
    "analysis": "Your analysis based on information provided",
    "services_with_configs": {
        "service1": ["./common_config", "./config1"],
        "service2": ["./common_config", "./config2", "./config3"],
        ......
    }
}
```
"""


class ProcessConfigCenterPrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _prompt),
        ("human", """# Structure of Configuration Center Source Directory
```
{{ dir_structure }}
```

# List of all service instances in this project
```
{{ services }}```""")], template_format="jinja2")
