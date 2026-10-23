import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# LangChain & Ollama imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import ConversationalRetrievalChain

app = FastAPI(title="Local RAG API", version="1.0")

# Configuration paths and models
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"

# In-memory session store for chat history: { "session_id": [(role, message), ...] }
chat_histories: Dict[str, List] = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


# Request body schemas using Pydantic
class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str


@app.post("/ingest")
def ingest_document(payload: IngestRequest):
    """Loads a markdown file, chunks it, and stores vectors in ChromaDB."""
    if not os.path.exists(payload.file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {payload.file_path}"
        )

    try:
        # 1. Load document
        loader = TextLoader(payload.file_path)
        docs = loader.load()

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # 3. Create embeddings & persist to ChromaDB
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {len(chunks)} chunks from {payload.file_path}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_with_docs(payload: ChatRequest):
    """Sends a question to the RAG pipeline with persistent chat history per session."""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Please /ingest a document first.",
        )

    try:
        # 1. Load vector store and retriever
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # 2. Setup LLM and Conversational Chain
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        # Initialize history for session if it doesn't exist
        if payload.session_id not in chat_histories:
            chat_histories[payload.session_id] = []

        # Conversational retrieval chain automatically manages chat history tuples
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        # 3. Execute chain with chat history
        # LangChain expects chat_history as a list of (human_msg, ai_msg) tuples
        response = qa_chain.invoke(
            {
                "question": payload.question,
                "chat_history": chat_histories[payload.session_id],
            }
        )

        # 4. Update session chat history
        chat_histories[payload.session_id].append(
            (payload.question, response["answer"])
        )

        return {
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": response["answer"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_rag(payload: QueryRequest):
    """Queries the local vector database and generates an answer using Ollama."""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Please /ingest a document first",
        )

    try:
        # 1. Load existing vector store
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
        )

        # 2. Setup QA retrieval chain
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        )

        # 3. Execute query
        response = qa_chain.invoke({"query": payload.question})

        return {"question": payload.question, "answer": response["result"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
def reset_vector_db():
    """Deletes the vector db if present"""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Please /ingest a document first",
        )
    shutil.rmtree(PERSIST_DIRECTORY)

    return {
        "status": "success",
        "message": f"Successfully deleted vector db {PERSIST_DIRECTORY}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
