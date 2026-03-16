from langchain_community.vectorstores import FAISS 
from src.models import embeddings_model
from pypdf import PdfReader

'''we can use also from langchain.text_splitter import RecursiveCharacterTextSplitter
and langchain's document class DirectoryLoader to load the file and split it into chunks '''

def read_file(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    with open(file_path, 'r') as file:
        return file.read()

def create_db(file_path):
    text = read_file(file_path)
    texts = [text[i:i+100] for i in range(0, len(text), 100)]
    vector_store = FAISS.from_texts(texts, embeddings_model)
    return vector_store
