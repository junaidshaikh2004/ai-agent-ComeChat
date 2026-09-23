from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.rag.loader import directory_loader

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def document_splitter() :
    documents = directory_loader()

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000 ,
        chunk_overlap =100
    )

    all_chunks = []
    for doc in documents :
        md_chunks = md_splitter.split_text(doc.page_content)
        for md_chunk in md_chunks :
            md_chunk.metadata.update(doc.metadata)

            pieces = recursive_splitter.split_text(md_chunk.page_content)
            for piece in pieces :
                all_chunks.append(Document(page_content=piece, metadata=md_chunk.metadata))

    print(f"chunked {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks