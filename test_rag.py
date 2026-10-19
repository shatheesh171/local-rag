import os
import shutil
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Clean up previous vector database run to avoid duplicate entries
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# 2. Load the generic markdown file
loader = TextLoader("./samp_knowledge.md")
docs = loader.load()

# 3. Split text into chunks (balanced for generic documentation)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 4. Create vector embeddings and store locally in ChromaDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
)

# 5. Set up the local retrieval chain with a lightweight model (Qwen 2.5 1.5B)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
)

# 6. Query the system
query = "What is platform engineering and what does it focus on?"
response = qa_chain.invoke({"query": query})

print("\n--- AI RESPONSE ---")
print(response["result"])
