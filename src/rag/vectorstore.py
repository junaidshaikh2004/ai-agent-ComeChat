from src.rag.splitter import document_splitter
from src.rag.embedder import get_embedder
from datetime import date
from langchain_chroma import Chroma
import os

# the meata data in chunks has datetime.time type values which cannot be soterd in chroma store , i.e we convert them to python string and then uild the vecore store
def build_vectorstore(chunks,embedder,persist_dir="chroma_store"):
    for chunk in chunks :
        for key,value in chunk.metadata.items():
            if isinstance(value, date):
                chunk.metadata[key] = value.isoformat()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir,
    )

    print(f"stored {len(chunks)} chunks in {persist_dir}")
    return vectorstore


#load the vectorstore

def load_vectorstore(persist_dir="chroma_store"):
    if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print("chroma store not built yet")
        return None

    vectorstore = Chroma(persist_directory=persist_dir,)
    return vectorstore