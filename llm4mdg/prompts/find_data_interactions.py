from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from .base import ChatPrompt, BasicPrompt

_analyze_prebuilt_service_prompt = """# Task Instructions
You are an AI assistant specializing in microservices architecture. Your task is to analyze pre-built services in an open-source microservices project. Based on the provided image name and port information, please complete the following tasks:

- Determine if the service is a well-known or common open-source service, and provide relevant background information.
- Identify the service's business type, including its function and purpose.
- Confirm the communication protocol used by the open ports (if applicable).
 
# Output Format
Your output should strictly obey the following JSON format. Call `ReturnResultTool` to return your result.
```
{
    "service": "identified common service name",   # if cannot find a common service, this value is set to null.
    "type": "the service's business type",         # if cannot determine the type of service, this value is set to null.
    "ports": [ {"port": PORT_NUMBER, "protocol": "Protocol"}, ... ],  # `protocol` can be set to null if cannot be inferred.
    "analysis": "explain why you reach the result"
}
```
"""

_analyze_non_prebuilt_service_prompt = """# Task Instructions
You are an AI assistant specializing in microservices architecture. Your task is to analyze non-prebuilt services in an open-source microservices project. You will receive a set of briefs of key code or configuration files related to the service. Based on this, complete the tasks below.

- Determine if the service uses a well-known or common open-source service, and provide relevant background information.
- Identify the service's business type, including its function and purpose.
- Confirm the communication protocol used by the open ports (if applicable).
- Find data interactions of this service, including ports, APIs, controllers, and routes that are exposed to external services, and files and code segments where this project proactively communicates with external services (e.g., consuming REST APIs, invoking SOAP services, using message queues).

# Output Format
Your output should strictly obey the following JSON format. Call `ReturnResultTool` to return your result.
```
{
    "analysis": "explain why you reach the result", # Keep this value clear and concise, within 100 words.
    "service": "identified common service name",   # if cannot find a common service, this value is set to null.
    "type": "the service's business type",         # if cannot determine the type of service, this value is set to null.
    "ports": [ {"port": PORT_NUMBER, "protocol": "Protocol"}, ... ],  # `protocol` can be set to null if cannot be inferred.
    "language": ["Mainly used programming languages in this service", ...]
    "interactions": [
        {
            "type": "Is this interaction exposes port(s) to passively accept requests from external sources, or actively send request to external sources. Only two values avaliable: `passive` and `active`.",
            "directionality": "The data flow direction of this interaction, only three values avaliable: `request-response`, `only-send`, `only-receive`.",
            "description": "Brief description of this interaction, keep it simple and concise.",
            "target_service": "The target service of this interaction. If interaction type is `passive`, this should be set to null.",
            "interaction_type": "Interaction methods or behaviour between microservices, Including but not limited to HTTP, WebSocket, TCP, UDP, RPC, Message Queue, Database Request, Cache Access, File Transfer, etc."
            "interaction_details": 
            # NOTICE: If you found multiple possible values for one interaction, just split them into multiple interactions. DON'T COMBINE THEM IN ONE INTERACTION OR SYNTAX ERROR WILL EASILY OCCUR!
            {
                # The content of this object depends on the type of interaction. Provide details about this interaction as key-value pairs.
                "method": "Use this attribute to store request method, like HTTP method, gRPC method, etc. Don't combine multiple method in one interaction, split them into different single interactions.",
                "host": "Use this attribute to store target host name or IP address. Don't store port number in this attribute.",
                
                # Use this attribute to store port number. If your value is not a valid number, use STRING rather than number to avoid syntax error.
                "port": PORT_NUMBER, 
                
                # OPTIONAL_ATTRS: If more details are found, feel free to add more attributes based on this interaction, as detailed as possible.
                "url": "Use this attribute to store URL, URI, routes-like data.",
                "queue_name": "...",
                "redis_command": "...",
                "database_name": "...",
                "query_arguments": "...",
                ...
            }
            
        }, ...
    ]
}
```
"""

_query_vector_db_prompt = """In this microservice project, please provide a comprehensive list of all the ports, APIs, controllers, and routes that are exposed to external services. Additionally, identify all the files and code segments where this project proactively communicates with external services (e.g., consuming REST APIs, invoking SOAP services, using message queues)."""

_validate_data_interactions_prompt = """# Task Instructions
You are an AI assistant specialized in microservices analysis. You will receive analysis results from another AI assistant for an open-source microservices project, formatted in JSON. Based on the provided analysis results, perform the following tasks:

- Remove all null values in the `interaction_details` dictionary under each interaction.
- Ensure and adjust the `host` and `port` values in `interaction_details` under each interaction, and make sure `host` and `port` are stored seperately. Note that PORT must be explicitly stated, even if it is special like 80 or 443. If the host contains unnecessary information, create and add corresponding entries in `interaction_details`. For example:
    ```
    # input
    - "interaction_details": { "host": "http://some-service:8000/abc/def" }
    - "interaction_details": { "host": "https://another-service" }

    # output
    - "interaction_details": { "host": "some-service", "port": 8000, "url": "/abc/def" }
    - "interaction_details": { "host": "another-service", "port": 443, "protocol": "https" }
    ```

# Output Formats
Your output should strictly obey the same JSON format as original analysis results. Call `ReturnResultTool` to return your result.
"""


class AnalyzePrebuiltServicePrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _analyze_prebuilt_service_prompt),
        ("human", """# Image name of target service
{{ image_name }}

# Open ports of this service in deployment config
{{ ports }}""")], template_format="jinja2")


class AnalyzeNonPrebuiltServicePrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _analyze_non_prebuilt_service_prompt),
        ("human", """# Service name
{{ service_name }}

# Open ports of this service in deployment config
{{ ports }}

# Retrieved briefs of key files
{{ rag_result }}
""")], template_format="jinja2")


class QueryVectorDBPrompt(BasicPrompt):
    prompt: ClassVar[PromptTemplate] = \
        PromptTemplate.from_template(_query_vector_db_prompt, template_format="jinja2")


class ValidateDataInteractionsPrompt(ChatPrompt):
    prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
        ("system", _validate_data_interactions_prompt),
        ("human", """# Current service name
{{ service_name }}

# Result parsed from LLM is:
```
{{ result }}
```""")], template_format="jinja2")
