from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import pymupdf  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
import re
from langchain_groq import ChatGroq

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

# Extract numerical score from LLM response
def extract_score(response):
    match = re.search(r'\b\d+\.\d+\b', response)
    if match:
        return float(match.group())
    return 0.0  # Default score if no number is found

# Validate chunk quality
def validate_chunk_quality(docs):
    if not docs:
        return {"error": "No documents to evaluate."}
    
    analysis_results = {
        'avg_chunk_length': sum(len(d.page_content) for d in docs) / len(docs),
        'coherence_score': [],
        'entity_continuity': []
    }
    
    llm = ChatGroq(model="llama3-8b-8192")
    
    coherence_prompt = """Analyze this text chunk for semantic coherence:
    {chunk}

    Provide a score from 0-1 (1=perfectly coherent) and a brief explanation:"""
    
    entity_prompt = """Check if these consecutive chunks maintain entity consistency:
    Chunk 1: {prev_chunk}
    Chunk 2: {current_chunk}

    Answer (Yes/No):"""
    
    for i, chunk in enumerate(docs[:10]):  # Evaluate first 10 chunks
        try:
            coherence_response = llm.invoke(coherence_prompt.format(chunk=chunk.page_content))
            coherence_score = extract_score(coherence_response.content)
            analysis_results['coherence_score'].append(coherence_score)

            if i > 0:
                response = llm.invoke(entity_prompt.format(
                    prev_chunk=docs[i-1].page_content,
                    current_chunk=chunk.page_content
                ))
                analysis_results['entity_continuity'].append(1 if "yes" in response.content.lower() else 0)

        except Exception as e:
            print(f"Error during validation: {e}")
            analysis_results['coherence_score'].append(0.0)
            if i > 0:
                analysis_results['entity_continuity'].append(0)
    
    return analysis_results

# Save embeddings to ChromaDB
def save_to_chromadb(docs, persist_directory="psyra_chromadb_baai"):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF_TOKEN"), model_name=EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    print("✔ Documents successfully saved to ChromaDB.")

if __name__ == "__main__":
    pdf_dir = "docs_Psyra"
    pdf_files = get_pdf_files(pdf_dir)
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    docs = process_pdfs(pdf_files)
    chunks = create_chunks(docs)
    
    print(f"🔹 Chunking completed. Total Chunks: {len(chunks)}")
    
    # Validate chunk quality
    validation_results = validate_chunk_quality(chunks)
    print(f"Chunk Quality Metrics: {validation_results}")
    
    save_to_chromadb(chunks)