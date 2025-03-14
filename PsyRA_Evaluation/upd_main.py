# upd_main.py

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Improved System Prompt
system_template = """
You are PsyRA, a highly skilled clinical psychologist specializing in structured intake interviews.

### **Guidelines**:
- Use **only** the retrieved context to answer.
- If no relevant context is found, respond with: "I don't have information on that."
- Keep responses **concise** and professional, with warmth.

### **History:** {history}
"""

# User Message Template
message_template = """
### **Context Retrieved**:
{context}

### **User Query**:
{question}

**Instructions**:
1. Respond based on the intake guidelines.
"""

# Load ChromaDB with optimized retrieval
def load_vectorstore():
    """Loads ChromaDB retriever for retrieving relevant context."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="psyra_chromadb_baai", embedding_function=embeddings)
    
    # Use Max Marginal Relevance (MMR) for better retrieval
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7})

# Retrieve relevant context from ChromaDB
def get_relevant_context(question, retriever):
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# Summarize conversation history dynamically
def summarize_conversation(history, force=False):
    if not history:
        return "No prior history."
    
    # Only summarize if history exceeds 5 turns
    if len(history) % 5 == 0 or force:
        summary_prompt = f"Summarize this conversation briefly:\n\n{history}"
        summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()
        
        try:
            return summarization_chain.invoke(summary_prompt)
        except Exception:
            return history[-500:]  # Fallback
    else:
        return history[-500:]  # Use recent 500 chars instead of summarizing

# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", message_template)])
    return prompt | llm | StrOutputParser()

# Save conversation data for evaluation
def save_conversation_data(conversations, filename="chatbot_responses.json"):
    try:
        # Load existing data if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
            
        # Update with new data
        for chatbot, data in conversations.items():
            if chatbot not in existing_data:
                existing_data[chatbot] = []
            existing_data[chatbot].extend(data)
            
        # Save updated data
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Conversation data saved to {filename}")
    except Exception as e:
        print(f"Error saving conversation data: {e}")

# Main chatbot loop
def main():
    retriever = load_vectorstore()
    chain = create_chat_chain()
    conversation_history = []
    
    # Data collection for evaluation
    chatbot_responses = {
        "PsyRA": []
    }

    print("Welcome to PsyRA. Type 'bye doctor' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nPsyRA: Thank you. Take care!")
            # Save conversation data before exiting
            save_conversation_data(chatbot_responses)
            break

        # Get relevant context from ChromaDB
        context = get_relevant_context(user_input, retriever)
        
        # Summarize conversation history
        summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

        # Prepare chain inputs
        inputs = {
            "context": context,
            "question": user_input,
            "history": summarized_history
        }

        # Get response with timing
        try:
            start_time = time.time()
            response = chain.invoke(inputs)
            response_time = time.time() - start_time
            
            print(f"\nPsyRA: {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
            # Store response data for evaluation
            chatbot_responses["PsyRA"].append({
                "response": response,
                "response_time": response_time,
                "user_input": user_input
            })

        except Exception as e:
            print(f"\nPsyRA: I apologize, but I'm having trouble processing that. Could you rephrase your question? (Error: {e})")

if __name__ == "__main__":
    main()