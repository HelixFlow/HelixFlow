![HelixFlow](docs/img/logo_1.png)

# HelixFlow

HelixFlow is a visual workflow builder for LangGraph-based agent applications. It provides a FastAPI backend for graph execution and a Umi-powered frontend for designing, testing, and debugging agent flows.

## Highlights

- Visual graph editor for agent workflows
- LangGraph-based execution engine
- Built-in nodes for LLM calls, conditional routing, and retrieval
- Annotation-based custom node development
- Test run and debugging workflow for inspecting graph execution
- MySQL-backed flow persistence
- Optional Milvus integration for vector retrieval

## Architecture

```text
HelixFlow
├── main.py              # FastAPI application entrypoint
├── router/              # HTTP API routes
├── core/                # Graph, node, field, and state abstractions
├── service/             # Runtime services
├── database/            # SQLModel models and database session setup
├── init.sql             # MySQL schema bootstrap
└── web/                 # Frontend application
```

## Requirements

- Python 3.12 recommended
- Node.js 18 recommended
- Yarn 1.x
- MySQL 8.x
- Milvus standalone, optional and only required for retrieval workflows

The backend Dockerfile currently uses Python 3.10, while local development has been tested with Anaconda Python 3.12.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/HelixFlow/HelixFlow.git
cd HelixFlow
git submodule update --init --recursive
```

### 2. Initialize MySQL

Start MySQL, then import the schema:

```bash
mysql -u root -p < init.sql
```

The schema creates the `helix` database and the required `flow` and `user` tables.

### 3. Start The Backend

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the database connection when your local credentials differ from the defaults:

```bash
export APP_TABLE_HOST=localhost
export APP_TABLE_PORT=3306
export APP_TABLE_USERNAME=root
export APP_TABLE_PASSWORD=123123
```

Run the API server:

```bash
python main.py
```

The backend listens on `http://127.0.0.1:11110` by default.

Health check:

```bash
curl http://127.0.0.1:11110/helixflow/health
```

### 4. Start The Frontend

```bash
cd web
yarn
yarn setup
HOST=127.0.0.1 PORT=8000 yarn dev
```

Open `http://127.0.0.1:8000`.

The frontend proxies `/helixflow` requests to `http://127.0.0.1:11110`.

## Configuration

Backend database settings can be configured with environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `APP_TABLE_HOST` | `localhost` | MySQL host |
| `APP_TABLE_PORT` | `3306` | MySQL port |
| `APP_TABLE_USERNAME` | `root` | MySQL username |
| `APP_TABLE_PASSWORD` | `123123` | MySQL password for local development |

Model and embedding nodes accept API key, model name, and base URL through node configuration at runtime. Do not commit real API keys or production credentials.

## Development Notes

- The backend entrypoint is `main.py`.
- The API prefix is `/helixflow`.
- The frontend app lives in `web/`.
- `web` is tracked as a submodule reference from the parent repository.
- Retrieval nodes require a running Milvus service and a prepared collection.

Recommended local startup order:

1. Start MySQL
2. Start Milvus if retrieval nodes are needed
3. Start the backend
4. Start the frontend

## Security

Keep the following out of Git:

- Real API keys
- Production database credentials
- JWT signing secrets
- Local IDE settings
- Build outputs and dependency directories
- Logs and cache files

Use environment variables for sensitive runtime configuration.

## Roadmap

- MCP node support
- Database agent nodes
- Business understanding nodes
- Flink SQL generation and job execution
- Image generation integrations

## Acknowledgements

This repo benefits from langgraph and langchain.
Special thanks to nextui for the amazing Logo design!

https://www.nextui.cc/#/home
