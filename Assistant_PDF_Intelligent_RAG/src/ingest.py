from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from src.models import embeddings_model

'''we can use also from langchain.text_splitter import RecursiveCharacterTextSplitter
and langchain's document class DirectoryLoader to load the file and split it into chunks '''


def read_file(file_path):
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # skip empty pages
            pages.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": i} 
            ))
    return pages

def create_db(file_paths):
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = []
    for file_path in file_paths:
        pages = read_file(file_path)           
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    return FAISS.from_documents(all_chunks, embeddings_model)
