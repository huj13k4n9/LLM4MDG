from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate

from .base import ChatPrompt

_identify_service_prompt = """As an expert in microservices architecture, you are tasked with analyzing an open-source microservices-based project. Your objective is to identify each service instance designed to run within this project and gather basic information about each service.

# Task Instructions

- Locate deploy configuration files (e.g., docker-compose.yml, Kubernetes yaml file, Vagrantfile, etc.), and detect the type of each deploy file (assumed only docker and kubernetes config type are legal, other types are not supported)
- Find all configuration files that should be used to deploy the whole project, analyze and identify every service instances in configuration files.
- For each service identified, you must justify why it is considered a microservice based on the provided sources, and distinguish each service as Prebuilt type or Non-Prebuilt type, which is described as follows.
    - Prebuilt: Some services like Redis, MongoDB, MySQL, RabbitMQ, etc. Typically these services don't need source code to build image, instead prebuilt images hosted in repositories (like Docker Hub) can be directly used for easier deployment.
    - Non-Prebuilt: Firstly, check if build directory is specified in deployment configuration files, or source code found in project directory; If not, try to discover the similarity or consistency between the project folder name and the service name. If a folder that can be associated is found, define it as Non-Probuilt.
- Identify every replica of images and recognize every replica as an unique service instance when each replica is for different use.
- If the instance is non-prebuilt type, find the directory containing its source code and data.
- Please identify the configuration file used only for a specific service based on the file name, and add it to the results of the corresponding service.
- Please identify common files used by all services, like gRPC protocol definitions, environment variables, etc.
- Please minimize the number of tool calls as much as possible to save tokens.

# Output Format

Your output should strictly obey the following JSON format. Once you get a full result of the task, Call `ReturnResultTool` with your result.
```
{
    "deploy_config": [{
        "path": "path of deploy configuration files",
        "type": "only 3 values available: `DOCKER`, `KUBERNETES`, `UNKNOWN`"
    }],
    "services": [{
            "name": "identifier of the service instance",
            "prebuilt": true,
            "evidence": "why you think this is a service instance",
            "configs": ["path of service-specific and common files"]
        },{
            "name": "identifier of the service instance",
            "prebuilt": false,
            "source_dir": "./path/to/directory/of/service"
            "evidence": "why you think this is a service instance",
            "configs": ["path of service-specific and common files"]
        }]
}
```

# An Example
```
# Deploy file
account-service:
    image: test/account-service
    ...
account-mongodb:
    image: test/mongodb
    ...
statistics-mongodb:
    image: test/mongodb
    ...
rabbitmq:
    image: rabbitmq:3
    ...

# Project directory
- account-service/  # Assume this directory contains codes
- mongodb/          # Assume this directory only contains data
- deploy.yaml       # This is the deploy config file

# Expected output
{
    "deploy_config": ["./deploy.yaml"],
    "services": [
        {
            "name": "account-service",
            "prebuilt": false,
            "evidence": "Service directory contains codes to build image.",
            "source_dir": "./account-service"
        }, {
            "name": "rabbitmq",
            "prebuilt": true,
            "evidence": "Uses prebuilt image and no codes found in project.",
        }, {
            "name": "statistics-mongodb",
            "prebuilt": true,
            "evidence": "Service directory only contains data, and uses prebuilt image."
        }, {
            "name": "account-mongodb",
            "prebuilt": true,
            "evidence": "Service directory only contains data, and uses prebuilt image."
        },
    ]
}
```"""


class IdentifyServicePrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _identify_service_prompt),
        ("human", "The path of project is {{ path }}"),
    ], template_format="jinja2")


_prompt = """# Task Instructions
You are an AI assistant specialized in microservices analysis. You will receive analysis results from another AI assistant for an open-source microservices project, formatted in JSON. Based on the provided analysis results and the project's directory structure, perform the following tasks:

- Verify the accuracy of the path information in the results. Correct any inaccuracies based on the directory structure.
- Convert any absolute paths to relative paths starting with `./` based on the project's root directory.

# Output Formats
Your output should strictly obey the following JSON format. Call `ReturnResultTool` to return your result.
```
{
    "modification": "Overall description of your modification on the original analysis results",
    "validated_result": { # Modified result in the same JSON format as original analysis results }
}
```
"""


class ValidateServicesPrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _prompt),
        ("human", """# Directory Structure of Microservice Project
```
{{ dir_structure }}
```
        
# Result parsed from LLM is:
```
{{ result }}
```""")], template_format="jinja2")
