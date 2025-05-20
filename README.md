# LLM4MDG

Related codes in paper \<LLM4MDG: Leveraging Large Language Model to Construct Microservices Dependency Graph\>, [Paper Link](https://ieeexplore.ieee.org/abstract/document/10944943/)

BibTeX Citation:

```bibtex
@INPROCEEDINGS{10944943,
  author={Hu, Jiekang and Li, Yakai and Xiang, Zhaoxi and Ma, Luping and Jia, Xiaoqi and Huang, Qingjia},
  booktitle={2024 IEEE 23rd International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
  title={LLM4MDG: Leveraging Large Language Model to Construct Microservices Dependency Graph},
  year={2024},
  volume={},
  number={},
  pages={859-869},
  keywords={Accuracy;Large language models;Scalability;Microservice architectures;Computer architecture;Knowledge graphs;Prompt engineering;Security;Software development management;Multi-agent systems;Microservice Architecture;Large Language Model;Prompt Engineering;Data Dependencies;Knowledge Graph},
  doi={10.1109/TrustCom63139.2024.00128}
}
```

## Tech Stack Used

- LLM framework: [LangChain](https://www.langchain.com/)
- Vector store: [Chroma](https://www.trychroma.com/) & [Milvus](https://milvus.io/)
- Logging: [Loguru](https://github.com/Delgan/loguru)
- Data Validation: [Pydantic](https://github.com/pydantic/pydantic)

## Basic Usage

First, install dependencies. Make sure you have Python >= 3.11 and [Poetry](https://github.com/python-poetry/poetry)
installed, and a running [Neo4j](https://neo4j.com/) instance available. Use the following command to install
dependencies, and Poetry will create a virtual environment for you.

```shell
$ poetry install --no-root
```

You can use docker to quickly deploy a Neo4j instance by simply typing the following command. For more details on
connection arguments, check the [Docker Hub page](https://hub.docker.com/_/neo4j).

```shell
$ docker run -d \
    -p 7474:7474 -p 7687:7687 \
    -v "$(pwd)/neo4j/data":/data \
    -v "$(pwd)/neo4j/logs":/logs \
    neo4j
```

Then, rename `config.yml.example` to `config.yml` and then edit it:

```yaml
# Whether to show the debug output of langchain.
debug: false

# The microservice project location that you want to analyze.
project_location: "path/to/microservice/project"

# If a config center is in the project, specify it.
# Please use identifier of config center service in the deployment file.
config_center_name: config
# Fill in the relative path of your specified project root directory.
config_center_dir: ./config

# Chat model configuration.
chat_model:
  type: "openai"  # "openai", "anthropic", "google" are supported.
  model: "gpt-4o-mini"
  base_url: "https://xxx/v1"
  api_key: "sk-"
  temperature: 0.0

# OpenAI Embedding model configuration.
openai_embedding:
  model: "text-embedding-ada-002"
  base_url: "https://xxx/v1"
  api_key: "sk-"

# Vector Database configuration.
# Currently, Milvus and Chroma are supported.
vector_db:
  # Milvus configuration.
  # Only `db_type`, `connection_type`, `connection_uri` are mandatory.
  db_type: "milvus"
  connection_type: "local"
  collection_name: "test"
  connection_uri: "./test.db"

  # Chroma configuration.
  #db_type: "chroma"
  #connection_type: "local"

# Neo4j configuration.
neo4j:
  # Connection URI (Required)
  uri: "neo4j://example.org:7687"
  # Authentication type (Required). Available values: "basic", "kerberos" and "bearer"
  auth_type: "basic"
  # This field is optional, default will connect "neo4j" database.
  database: "neo4j"

  # Options for "basic"
  username: "username"
  password: "password"

  # Options for "kerberos"
  kerberos_ticket: "base64_encoded_ticket"

  # Options for "bearer"
  bearer_token: "base64_encoded_token"
```

Last, run `main.py` to execute the analysis procedure.

```shell
$ python main.py
```

Or, you can directly import LLM4MDG to run.

```python
from llm4mdg import LLM4MDG

if __name__ == "__main__":
    LLM4MDG.from_config(file_path="path/to/config/yml").run()
```
