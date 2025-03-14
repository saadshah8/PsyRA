import pymupdf as fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Function to get all PDF files from a directory
def get_pdf_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]

# PDF Processing
def process_pdf(pdf_paths):
    documents = []
    for path in pdf_paths:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        documents.append(Document(page_content=text, metadata={"source": path}))
    return documents

# Create text chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Save embeddings to ChromaDB
def save_to_chromadb(docs, persist_directory="disorders_chroma_vectoredb"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    print("Documents saved to ChromaDB.")

if __name__ == "__main__":
    pdf_dir = "docs_Psyra"  # Directory containing PDFs
    pdf_files = get_pdf_files(pdf_dir)  # Get all PDF files
    docs = process_pdf(pdf_files)
    chunks = create_chunks(docs)
    save_to_chromadb(chunks)

