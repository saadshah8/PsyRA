from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pymupdf as fitz
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# PDF Processing
def process_pdf(pdf_path):
    documents = []
    for path in pdf_path:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        documents.append(Document(page_content=text))
    return documents

# Create text chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Initialize embeddings and vector store
def initialize_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Summarize previous conversation history
def summarize_conversation(history):
    if not history:
        return "No previous context"

    summary_prompt = f"Summarize the following conversation briefly:\n\n{history}"
    summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()
    
    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return history[-500:]  # Fallback: Keep last 500 characters if summarization fails

# Retrieve relevant context
def get_relevant_context(question, retriever):
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# System Prompt Template
system_template = """
You are Dr. PsyRA, a clinical psychologist specializing in intake interviews..

Key responsibilities:
1. Conduct structured assessments
2. Perform risk evaluations
3. Gather background information
4. Maintain warm and empathetic communication

Guidelines:
- Use ONLY information from the provided context
- If information isn't in the context, say "I don't have specific guidelines on that"
- Keep responses **short and to the point** for better understanding
- Maintain professional yet warm tone

Current conversation stage: {stage}
Previous context: {history}
"""

# User Message Template
message_template = """
Context: {context}

User Question: {question}

Respond based on intake guidelines while maintaining a warm, professional tone.
"""

# Define conversation stages
stages = {
    "initial": "Initial Engagement",
    "background": "Background Information",
    "risk": "Risk Assessment",
    "mental_health": "Current Presentation",
    "family": "Family History",
    "planning": "Care Planning"
}

# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", message_template)
    ])
    return prompt | llm | StrOutputParser()

# Main function
def main():
    pdf_files = ["docs_Psyra/PTSD.pdf","docs_Psyra/ADHD.pdf"]
    docs = process_pdf(pdf_files)
    chunks = create_chunks(docs)
    retriever = initialize_vectorstore(chunks)
    
    chain = create_chat_chain()
    conversation_history = []
    current_stage = "initial"
    
    print("Welcome to the Clinical Intake Assistant. Type 'exit' to end the session.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA: Thank you for the session. Take care!")
            break

        # Get relevant context
        context = get_relevant_context(user_input, retriever)

        # Summarize conversation history
        summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

        # Prepare chain inputs
        inputs = {
            "context": context,
            "question": user_input,
            "stage": stages[current_stage],
            "history": summarized_history
        }

        # Get response
        try:
            response = chain.invoke(inputs)
            print(f"\nDr. PsyRA: {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")

            # Progress stage if needed
            if current_stage == "initial" and "background" in user_input.lower():
                current_stage = "background"
            elif current_stage == "background" and "risk" in user_input.lower():
                current_stage = "risk"
            
        except Exception as e:
            print("\nDr. PsyRA: I apologize, but I'm having trouble processing that. Could you rephrase your question?")

if __name__ == "__main__":
    main()



# ===================================================================================================================


# import pymupdf as fitz  # PyMuPDF
# import os
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# # Load environment variables
# load_dotenv()
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # PDF Processing
# def process_pdf(pdf_path):
#     documents = []
#     for path in pdf_path:
#         doc = fitz.open(path)
#         text = "\n".join([page.get_text() for page in doc])
#         documents.append(Document(page_content=text, metadata={"source": path}))
#     return documents

# # Create text chunks
# def create_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return text_splitter.split_documents(documents)

# # Save embeddings to ChromaDB
# def save_to_chromadb(docs, persist_directory="disorders_chroma_vectoredb"):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
#     #vectorstore.persist()  # Save database to disk
#     print("Documents saved to ChromaDB.")

# if __name__ == "__main__":
#     pdf_files = ["docs_Psyra/PTSD.pdf", "docs_Psyra/ADHD.pdf"]  # Add more as needed
#     docs = process_pdf(pdf_files)
#     chunks = create_chunks(docs)
#     save_to_chromadb(chunks)


