import pymupdf as fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import re

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

# Extract numerical score from LLM response
def extract_score(response):
    # Use regex to find a float in the response
    match = re.search(r'\b\d+\.\d+\b', response)
    if match:
        return float(match.group())
    return 0.0  # Default score if no number is found

# Validate chunk quality
def validate_chunk_quality(docs):
    """
    Evaluates text splitting effectiveness without labeled data.
    """
    if not docs:
        return {"error": "No documents to evaluate."}

    analysis_results = {
        'avg_chunk_length': sum(len(d.page_content) for d in docs) / len(docs),
        'coherence_score': [],
        'entity_continuity': []
    }

    # Define LLM and prompts
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
            # Coherence analysis
            coherence_response = llm.invoke(coherence_prompt.format(chunk=chunk.page_content))
            coherence_score = extract_score(coherence_response.content)
            analysis_results['coherence_score'].append(coherence_score)

            # Entity continuity
            if i > 0:
                response = llm.invoke(entity_prompt.format(
                    prev_chunk=docs[i-1].page_content,
                    current_chunk=chunk.page_content
                ))
                analysis_results['entity_continuity'].append(1 if "yes" in response.content.lower() else 0)

        except Exception as e:
            print(f"Error during validation: {e}")
            analysis_results['coherence_score'].append(0.0)  # Append default score on error
            if i > 0:
                analysis_results['entity_continuity'].append(0)  # Append default continuity score on error
    
    return analysis_results

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

    # Validate chunk quality
    validation_results = validate_chunk_quality(chunks)
    print(f"Chunk Quality Metrics: {validation_results}")

    # Save processed chunks to ChromaDB
    save_to_chromadb(chunks)