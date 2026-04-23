![header](docs/img/logo_1.png)

# HelixFlow

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10-blue)

> HelixFlow is a visual LangGraph agent builder


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
- [x] Retrieval (embeddings, vector store)
- [ ] Business understanding
- [ ] Db agent
- [ ] Flink SQL Generator
- [ ] Flink Job Runner
- [ ] Image Generation (Stable Diffusion ComfyUI)

## 🚀 Quick start

----


```
    # step 1: Clone the repository
    git clone https://github.com/HelixFlow/HelixFlow.git
    git submodule update --init --recursive
    # step 2: start the backend
    cd HelixFlow
    docker build -t helixflow .
    docker run -d -p 11110:11110 helixflow
    # step 3: start the frontend
    cd web
    docker build -t helixflow-web .
    docker run -d -p 8000:8000 helixflow-web
    # step 4: open the browser and go to http://localhost:8000


```


## ⚠️ Known Limitations (P0'-a)

This release focuses on stabilising the core runtime. The following items are **intentionally deferred**:

- **Retrieval nodes are placeholder skeletons.** Modules under `core/builtin/retrieval/` are
  scaffolding only; no runtime code path imports `pymilvus` or `langchain_milvus`. A complete
  implementation is scheduled for **P3'** (see RFC-2026-001 §7.1 and §3.3 B7).
  If you want to experiment ahead of P3', install the optional extras:
  ```bash
  pip install -r requirements-optional.txt
  ```
- **LangGraph is pinned at 1.0.2.** The upgrade to 1.1.9 (and the switch to a PostgresSaver
  checkpointer) is scheduled for **P0'-b Sprint 2**.
- **Persistence layer is MySQL-only** for P0'-a. PostgreSQL + pgvector migration is a P0'-b
  Track-1 deliverable.
- **SSE / streaming flow execution** endpoints are wired as skeletons only; the first streaming
  sprint is P0'-b (see `tests/api/test_flow_process_sse_event_order.py` — currently skipped).

See [RFC-2026-001](docs/rfc-helixflow-rt-agent-master-architecture.md) for the full roadmap.


## 🙏 Acknowledgement

----
This repo benefits from **[langgraph](https://github.com/langchain-ai/langgraph)** and **[langchain](https://github.com/langchain-ai/langchain)**.
Special thanks to **[nextui](https://www.nextui.cc)** for the amazing Logo design!


## License

Licensed under the [Apache License 2.0](./LICENSE).

Copyright 2025-2026 Mango, begc, and HelixFlow Contributors.
