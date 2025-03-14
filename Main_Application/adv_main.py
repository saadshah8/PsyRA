# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_message_histories import ChatMessageHistory

# import os
# import re
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# groq_api = os.getenv("GROQ_API_KEY")

# # Ensure API Key is available
# if not groq_api:
#     raise ValueError("Missing GROQ_API_KEY. Please check your .env file.")

# # Use a more advanced embedding model
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# # Initialize conversation history
# history = ChatMessageHistory()

# # Optimized Memory
# memory = ConversationBufferMemory(
#     memory_key="history",
#     return_messages=True,
#     chat_memory=history
# )

# # Conversation Stages
# stages = {
#     "initial": "Initial Engagement",
#     "background": "Background Information",
#     "risk": "Risk Assessment",
#     "mental_health": "Current Presentation",
#     "family": "Family History",
#     "planning": "Care Planning"
# }

# # System Prompt Template
# system_template = """
# You are Dr. PsyRA, a clinical psychologist specializing in intake interviews.

# ### Key Responsibilities:
# 1. Conduct structured assessments
# 2. Perform risk evaluations
# 3. Gather background information
# 4. Maintain warm and empathetic communication

# ### **Current Stage: {stage}**
# - Focus on relevant details for this stage.
# - Keep responses **brief and professional**.

# **Summarized Conversation History:**  
# {history}
# """

# # User Message Template
# message_template = """
# ### Context:
# {context}

# ### User Question:
# {question}

# ### Instructions:
# 1. Answer concisely with relevant clinical insights.
# 2. If needed, suggest transitioning to another stage.
# """

# # Load ChromaDB with optimized search
# def load_vectorstore():
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
#     vectorstore = Chroma(
#         persist_directory="disorders_chroma_vectoredb_adv",
#         embedding_function=embeddings
#     )
    
#     return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

# # Retrieve relevant context from ChromaDB
# def get_relevant_context(question, retriever):
#     docs = retriever.invoke(question)
#     return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# # Create chat chain
# def create_chat_chain():
#     llm = ChatGroq(model="llama3-8b-8192")
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_template),
#         ("human", message_template)
#     ])
#     return prompt | llm | StrOutputParser()

# # Chatbot loop
# def main():
#     retriever = load_vectorstore()
#     chain = create_chat_chain()
#     current_stage = "initial"

#     print("Welcome to the Clinical Intake Assistant. Type 'bye doctor' to end the session.")

#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() == "bye doctor":
#             print("\nDr. PsyRA: Thank you for the session. Take care!")
#             break

#         try:
#             context = get_relevant_context(user_input, retriever)
#             summarized_history = memory.load_memory_variables({})["history"]

#             inputs = {
#                 "context": context,
#                 "question": user_input,
#                 "stage": stages[current_stage],
#                 "history": summarized_history
#             }

#             response = chain.invoke(inputs)

#             # Check for stage change
#             stage_match = re.search(r"\*\*\[Next Stage: (.*?)\]\*\*", response)
#             if stage_match:
#                 new_stage = stage_match.group(1).strip().lower()
#                 if new_stage in stages:
#                     current_stage = new_stage
#                     print(f"\n(Stage updated to: {stages[current_stage]})")

#             response = re.sub(r"\*\*\[Next Stage: .*?\]\*\*", "", response).strip()
#             print(f"\nDr. PsyRA: {response}")

#             # Save conversation history
#             memory.save_context({"input": user_input}, {"output": response})

#         except Exception as e:
#             print("\nDr. PsyRA: I'm having trouble processing that. Could you rephrase?")
#             print(f"Error: {e}")

# if __name__ == "__main__":
#     main()

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Improved System Prompt
system_template = """
You are Dr. PsyRA, a highly skilled clinical psychologist specializing in structured intake interviews.

### **Guidelines**:
- Use **only** the retrieved context to answer.
- If no relevant context is found, respond with: "I don't have information on that."
- Keep responses **concise** and professional, with warmth.
- Adapt answers based on the **conversation stage**.

### **Current Stage: {stage}**
- **History:** {history}
"""

# User Message Template
message_template = """
### **Context Retrieved**:
{context}

### **User Query**:
{question}

**Instructions**:
1. Respond based on the intake guidelines.
2. Detect if the conversation should **transition** to another stage, but dont mention it in the response to user

"""
# 3. If a stage change is needed, **mark it** clearly in your response.


# Load ChromaDB with optimized retrieval
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="psyra_chromadb_baai", embedding_function=embeddings)
    
    # Use Max Marginal Relevance (MMR) for better retrieval
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

# Retrieve relevant context
def get_relevant_context(question, retriever):
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# Summarize conversation history dynamically
def summarize_conversation(history):
    if not history:
        return "No prior history."

    summary_prompt = f"Summarize this conversation briefly:\n\n{history}"
    summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()

    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return history[-500:]  # Fallback to last 500 characters

# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", message_template)])
    return prompt | llm | StrOutputParser()

# Main chatbot loop
def main():
    retriever = load_vectorstore()
    chain = create_chat_chain()
    conversation_history = []
    current_stage = "initial"

    print("Welcome to Dr. PsyRA. Type 'bye doctor' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA: Thank you. Take care!")
            break

        context = get_relevant_context(user_input, retriever)
        summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

        inputs = {"context": context, "question": user_input, "stage": current_stage, "history": summarized_history}

        response = chain.invoke(inputs)

        stage_match = re.search(r"\*\*\[Next Stage: (.*?)\]\*\*", response)
        if stage_match:
            current_stage = stage_match.group(1).strip().lower()

        print(f"\nDr. PsyRA: {response}")

        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Assistant: {response}")

if __name__ == "__main__":
    main()
