import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain & Ollama imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

app = FastAPI(title="Local RAG API", version="1.0")

# Configuration paths and models
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"


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
