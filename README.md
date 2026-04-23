![header](docs/img/logo_1.png)

# HelixFlow

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10-blue)

> HelixFlow is a visual LangGraph agent builder.

## ✨ Features

----
**1. Visual Agent Builder**
- Drag and drop to create agents
- Visualize agent flow with nodes and edges

**2. Customizable Nodes**
- Create custom nodes with annotations

**3. Built-in Nodes**
- [X] LLM
- [X] if condition
- [ ] MCP
- [x] Retrieval (embeddings, vector store) — **placeholder skeleton only in P0'-a, see Known Limitations below**
- [ ] Business understanding
- [ ] Db agent
- [ ] Flink SQL Generator
- [ ] Flink Job Runner
- [ ] Image Generation (Stable Diffusion ComfyUI)

## 🚀 Quick start

----

```shell
# step 1: Clone the repository
git clone https://github.com/HelixFlow/HelixFlow.git
cd HelixFlow
git submodule update --init --recursive

# step 2: bring up the backend stack (api + mysql) via docker compose
docker compose up -d --wait
# verify: /helixflow/health should return {"status": "OK"}
curl http://localhost:11110/helixflow/health

# step 3: start the frontend (separate repo / submodule)
cd web
docker build -t helixflow-web .
docker run -d -p 8000:8000 helixflow-web

# step 4: open the browser and go to http://localhost:8000
```

### Configuration

The API reads its MySQL connection from these environment variables
(consumed by `config/app_config.py`; the compose file sets them for you):

| Var                    | Default      | Notes                                             |
|------------------------|--------------|---------------------------------------------------|
| `APP_TABLE_HOST`       | `localhost`  | Hostname/IP of MySQL — compose sets it to `mysql` |
| `APP_TABLE_PORT`       | `3306`       | Container port. Host maps to `13306` externally   |
| `APP_TABLE_USERNAME`   | `root`       |                                                   |
| `APP_TABLE_PASSWORD`   | `123123`     |                                                   |

> Host port `13306` is used to avoid colliding with a locally-running
> Homebrew MySQL on `3306`. Inside the compose network the service is still
> reachable as `mysql:3306`.

## ⚠️ Known Limitations (P0'-a)

P0'-a is the "bug-stop + ship-ready gate" milestone. Several subsystems ship
as skeletons for now; the timeline is tracked in RFC-2026-001 §7.1.

- **Retrieval nodes** (`core/builtin/retrieval/`) are placeholder skeletons.
  No runtime code path imports `pymilvus` or `langchain_milvus`. Real
  implementation is scheduled for P3' (see RFC-2026-001 §3.3 B7).
  Install `pip install -r requirements-optional.txt` only if you plan to
  experiment with retrieval ahead of P3'.
- **Checkpointer backends**: only `memory` (LangGraph `InMemorySaver`) is
  fully supported. Passing `sqlite` or `postgres` to `compile_graph` logs a
  warning and falls back to memory. Real PostgresSaver + pgvector support
  lands in P0'-b (RFC §7.1 P0'-b Track-1).
- **SSE streaming**: `POST /helixflow/flows/process` currently returns the
  result synchronously as JSON. The SSE event stream is landing in P0'-b
  Sprint 2 (skeleton test at `tests/api/test_flow_process_sse_event_order.py`).
- **LangGraph** is pinned to `1.0.2` for P0'-a; upgrade to `1.1.9` is
  queued for P0'-b (file `tests/smoke/test_langgraph_119_import_and_invoke.py`
  already carries the "119" marker for the upgrade gate).
- **Persistence layer is MySQL-only** for P0'-a. PostgreSQL + pgvector
  migration is a P0'-b Track-1 deliverable.

See [RFC-2026-001](../knowledge/research/rfc-2026-04-23-helixflow-rt-agent-master-architecture.md) for the full roadmap.

## 🙏 Acknowledgement

----
This repo benefits from **[langgraph](https://github.com/langchain-ai/langgraph)**
and **[langchain](https://github.com/langchain-ai/langchain)**.
Special thanks to **[nextui](https://www.nextui.cc)** for the amazing Logo design!

## License

Licensed under the [Apache License 2.0](./LICENSE).

Copyright 2025-2026 Mango, begc, and HelixFlow Contributors.
