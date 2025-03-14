# import os
# import pymupdf as fitz  # PyMuPDF
# import asyncio
# from dotenv import load_dotenv
# from concurrent.futures import ThreadPoolExecutor
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


# # Load environment variables
# load_dotenv()
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # Use a more advanced embedding model for better vector search
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# # Function to get all PDF files from a directory
# def get_pdf_files(directory):
#     return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]

# # Function to extract text from a single PDF
# def extract_text_from_pdf(pdf_path):
#     try:
#         doc = fitz.open(pdf_path)
#         text = "\n".join(page.get_text() for page in doc)
#         return Document(page_content=text, metadata={"source": pdf_path})
#     except Exception as e:
#         print(f"Error processing {pdf_path}: {e}")
#         return None

# # Process multiple PDFs concurrently
# async def process_pdfs_async(pdf_paths):
#     loop = asyncio.get_running_loop()
#     with ThreadPoolExecutor() as executor:
#         tasks = [loop.run_in_executor(executor, extract_text_from_pdf, path) for path in pdf_paths]
#         results = await asyncio.gather(*tasks)
#     return [doc for doc in results if doc]

# # Create text chunks
# def create_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return text_splitter.split_documents(documents)

# # Save embeddings to ChromaDB
# def save_to_chromadb(docs, persist_directory="disorders_chroma_vectoredb_adv"):
#     embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF_TOKEN"), model_name=EMBEDDING_MODEL)


#     vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
#     print("Documents saved to ChromaDB.")

# async def main():
#     pdf_dir = "docs_Psyra"  
#     pdf_files = get_pdf_files(pdf_dir)
#     docs = await process_pdfs_async(pdf_files)
#     chunks = create_chunks(docs)
#     save_to_chromadb(chunks)

# if __name__ == "__main__":
#     asyncio.run(main())


from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import pymupdf  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Use a more powerful embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Get all PDF files from a directory
def get_pdf_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]

# Process a single PDF file
def process_single_pdf(path):
    doc = pymupdf.open(path)
    text = "\n".join([page.get_text("text").strip() for page in doc if page.get_text()])
    text = text.replace("\n", " ").strip()  # Normalize text formatting
    return Document(page_content=text, metadata={"source": os.path.basename(path)})

# Process PDFs using threading
def process_pdfs(pdf_paths):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(process_single_pdf, pdf_paths))

# Create text chunks with adaptive chunking
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    return text_splitter.split_documents(documents)

# Save embeddings to ChromaDB
def save_to_chromadb(docs, persist_directory="psyra_chromadb_baai"):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF_TOKEN"), model_name=EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    print("✔ Documents successfully saved to ChromaDB.")

if __name__ == "__main__":
    pdf_dir = "docs_Psyra"
    pdf_files = get_pdf_files(pdf_dir)
    
    #Debug
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    docs = process_pdfs(pdf_files)
    chunks = create_chunks(docs)
    
    print(f"🔹 Chunking completed. Total Chunks: {len(chunks)}")
    
    save_to_chromadb(chunks)
