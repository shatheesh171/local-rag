# Local Privacy-First RAG Pipeline & Chat Service

A lightweight, fully offline Retrieval-Augmented Generation (RAG) backend service built with Python, FastAPI, LangChain, ChromaDB, and Ollama. 

Designed for data privacy and local execution, this project eliminates external API dependencies by running all embeddings and LLM inferences entirely on local hardware.

---

## 🏗️ Architecture & Tech Stack
- **Orchestration:** LangChain (Text splitting, Retrieval chains, Conversational memory)
- **Vector Database:** ChromaDB (Persistent local embedding storage)
- **Local LLM & Embeddings:** Ollama (`qwen2.5:1.5b` for text generation, `nomic-embed-text` for vector embeddings)
- **Web API Framework:** FastAPI & Pydantic (REST endpoints with interactive Swagger docs)

---

## 📂 Project Structure
```text
local-rag-demo/
│
├── sample_knowledge.md   # Generic markdown knowledge base
├── test_rag.py           # Core RAG pipeline script
├── app.py                # FastAPI backend service (Ingest, Query, Chat)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

🚀 Getting Started
1. Prerequisites
Ensure Ollama is installed and running locally.

Pull the required local models:

Bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
2. Setup Environment
Bash
git clone [https://github.com/shatheesh171/local-rag.git](https://github.com/shatheesh171/local-rag.git)
cd local-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Run the Standalone Script
To test the foundational RAG execution engine:

Bash
python test_rag.py
4. Run the FastAPI Web Service
To start the REST API server:

Bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
Access the interactive OpenAPI/Swagger UI at http://127.0.0.1:8000/docs.

📌 API Endpoints & Usage
1. Ingest Knowledge Document
Chunks and stores text embeddings from a local file into ChromaDB.

Bash
curl -X POST "[http://127.0.0.1:8000/ingest](http://127.0.0.1:8000/ingest)" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./sample_knowledge.md"}'
2. Stateless Query
Performs semantic vector retrieval and context-driven generation without history.

Bash
curl -X POST "[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is platform engineering?"}'
3. Conversational Session Chat
Maintains stateful memory using a session ID for multi-turn dialogue.

Bash
curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "session-101",
           "question": "What are its main responsibilities?"
         }'
🛡️ Key Features & RAG Guardrails
100% Offline Data Privacy: Local vector index storage and model inference prevent external telemetry or data leaks.

Strict Context Boundaries: Prompt templates constrain the LLM to ground answers in retrieved chunks, mitigating hallucinations.

Persistent Vector Storage: Chunks are indexed into ChromaDB with metadata for fast nearest-neighbor retrieval.

Multi-Turn Session Memory: Tracks chat history across turns, automatically rephrasing follow-up queries for context retrieval.

🔮 Future Enhancements
[ ] Add PDF document support (PyPDFLoader).

[ ] Implement chunk metadata filtering and source citation attribution.

[ ] Incorporate hybrid search (keyword BM25 + dense vector embeddings).


