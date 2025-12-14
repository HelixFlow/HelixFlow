![header](docs/img/logo_1.png)
# HelixFlow
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


## 🙏 Acknowledgement

----
This repo benefits from **[langgraph](https://github.com/langchain-ai/langgraph)** and **[langchain](https://github.com/langchain-ai/langchain)**.  
Special thanks to **[nextui](https://www.nextui.cc)** for the amazing Logo design!  