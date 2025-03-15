from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import pymupdf  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
import re
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Keep the original embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Define pages to skip (irrelevant sections)
SKIP_PAGES = list(range(1, 41))  # Skips intro, legal disclaimers, and acknowledgments

# Get all PDF files from a directory
def get_pdf_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]

# Extract disorder name from section titles (used for metadata)
def extract_disorder_name(text):
    lines = text.split("\n")
    for line in lines:
        if "Disorder" in line or "Syndrome" in line or "Diagnosis" in line:
            return line.strip()
    return "General"

# Process a single PDF file (extract relevant content, remove unnecessary sections)
def process_single_pdf(path):
    doc = pymupdf.open(path)
    texts = []
    metadata_list = []

    for i, page in enumerate(doc):
        if i + 1 in SKIP_PAGES:
            continue  # Skip predefined irrelevant pages
        
        text = page.get_text("text").strip()
        if len(text) > 20:  # Ignore empty/meaningless pages
            disorder_name = extract_disorder_name(text)
            texts.append(Document(page_content=text, metadata={"disorder": disorder_name, "source": os.path.basename(path)}))

    return texts

# Process PDFs using threading
def process_pdfs(pdf_paths):
    with ThreadPoolExecutor() as executor:
        docs = executor.map(process_single_pdf, pdf_paths)
    return [doc for sublist in docs for doc in sublist]  # Flatten the list

# Create text chunks with disorder names as metadata
def create_chunks(documents, min_chunk_size=100):
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)
    
    # Remove empty or too short chunks
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip() and len(chunk.page_content) >= min_chunk_size]
    
    return valid_chunks

# Remove duplicate chunks to avoid redundant embeddings
def remove_duplicates(chunks):
    seen = set()
    unique_chunks = []

    for chunk in chunks:
        if not chunk.page_content.strip():  # Skip empty chunks
            continue
        
        content_hash = hash(chunk.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

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
def save_to_chromadb(docs, persist_directory="chroma_book_baai"):
    # embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF_TOKEN"), model_name=EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Filter out empty documents
    docs = [doc for doc in docs if doc.page_content.strip()]

    if not docs:
        print("❌ No valid documents to store in ChromaDB.")
        return

    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    print("✔ DSM-5-TR content successfully saved to ChromaDB.")


if __name__ == "__main__":
    pdf_dir = "bookk"
    pdf_files = get_pdf_files(pdf_dir)

    print(f"📂 Found {len(pdf_files)} PDF files. Processing DSM-5-TR...")

    docs = process_pdfs(pdf_files)
    print(f"✅ Processing completed. Total Documents: {len(docs)}")

    chunks = create_chunks(docs)
    print(f"🔹 Chunking completed. Total Chunks before filtering: {len(chunks)}")

    chunks = remove_duplicates(chunks)
    print(f"✅ Final Chunk Count after removing duplicates: {len(chunks)}")
    
    # Validate chunk quality
    validation_results = validate_chunk_quality(chunks)
    print(f"Chunk Quality Metrics: {validation_results}")

    save_to_chromadb(chunks)