from src.rag.loader import directory_loader
from src.rag.splitter import document_splitter
from src.rag.embedder import get_embedder
from src.rag.vectorstore import build_vectorstore,load_vectorstore
def main():
    # directory_loader()
    # document_splitter()
    # document_embedder()
    build_vectorstore(chunks=document_splitter(),embedder=get_embedder(),persist_dir="chroma_store")
    # load_vectorstore(persist_dir="chroma_store")
    
if __name__ == "__main__":
    main()
