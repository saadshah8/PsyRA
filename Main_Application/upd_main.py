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

# Updated System Prompt (Stages Removed)
system_template = """
You are Dr. PsyRA, a highly skilled clinical psychologist specializing in structured intake interviews.

### **Guidelines**:
- Provide professional, empathetic, and structured responses.
- If the user describes symptoms, explain them in a clear and supportive way.
- Offer coping strategies where relevant.
- If unsure about something, respond naturally without stating "I don't have information on that."
- Keep responses **Short** and **concise** mainly and professional, but warm and reassuring.

"""


# User Message Template
message_template = """
### **User Query**:
{question}

**Guidelines for Response**:
- Offer insights based on known psychological knowledge.
- Address the user's concerns with empathy and clarity.
- If symptoms align with a condition, provide an explanation in layman's terms and suggest practical coping strategies.
- Avoid unnecessary technical jargon unless the user asks for specifics.
"""


# Load ChromaDB with optimized retrieval
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="chroma_book_baai", embedding_function=embeddings)
    
    # Use Max Marginal Relevance (MMR) for better retrieval
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7})

# Retrieve relevant context
def get_relevant_context(question, retriever):
    docs = retriever.invoke(question)
    
    if not docs:
        return "No relevant context found."
    
    # Extract disorder names from metadata
    top_disorders = list(set([doc.metadata.get("disorder", "General") for doc in docs]))
    
    context_text = "\n".join([doc.page_content for doc in docs])
    return f"**Possible Relevant Disorders:** {', '.join(top_disorders)}\n\n{context_text}"

# # Summarize conversation history dynamically
# def summarize_conversation(history, force=False):
#     if not history:
#         return "No prior history."
    
#     # Only summarize if history exceeds 5 turns
#     if len(history) % 5 == 0 or force:
#         summary_prompt = f"Summarize this conversation briefly:\n\n{history}"
#         summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()
        
#         try:
#             return summarization_chain.invoke(summary_prompt)
#         except Exception:
#             return history[-500:]  # Fallback
#     else:
#         return history[-500:]  # Use recent 500 chars instead of summarizing
    
def summarize_conversation(history):
    if not history:
        return "No previous context"

    summary_prompt = f"Summarize the following conversation briefly:\n\n{history}"
    summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()

    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return history[-500:]  # Fallback: Keep last 500 characters if summarization fails

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

    print("Welcome to Dr. PsyRA. Type 'bye doctor' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA: Thank you. Take care!")
            break

        context = get_relevant_context(user_input, retriever)
        summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

        inputs = {"context": context, "question": user_input, "history": summarized_history}

        response = chain.invoke(inputs)

        print(f"\nDr. PsyRA: {response}")

        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Assistant: {response}")

if __name__ == "__main__":
    main()






# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import os
# import re
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# groq_api = os.getenv("GROQ_API_KEY")

# # Improved System Prompt
# system_template = """
# You are Dr. PsyRA, a highly skilled clinical psychologist specializing in structured intake interviews.

# ### **Guidelines**:
# - Use **only** the retrieved context to answer.
# - If no relevant context is found, respond with: "I don't have information on that."
# - Keep responses **concise** and professional, with warmth.
# - Adapt answers based on the **conversation stage**.

# ### **Current Stage: {stage}**
# - **History:** {history}
# """

# # User Message Template
# message_template = """
# ### **Context Retrieved**:
# {context}

# ### **User Query**:
# {question}

# **Instructions**:
# 1. Respond based on the intake guidelines.
# 2. Detect if the conversation should **transition** to another stage, but dont mention it in the response to user

# """
# # 3. If a stage change is needed, **mark it** clearly in your response.


# # Load ChromaDB with optimized retrieval
# def load_vectorstore():
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
#     vectorstore = Chroma(persist_directory="psyra_chromadb_baai", embedding_function=embeddings)
    
#     # Use Max Marginal Relevance (MMR) for better retrieval
#     return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7})


# # Retrieve relevant context
# def get_relevant_context(question, retriever):
#     docs = retriever.invoke(question)
#     return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

# # Summarize conversation history dynamically
# def summarize_conversation(history, force=False):
#     if not history:
#         return "No prior history."
    
#     # Only summarize if history exceeds 5 turns
#     if len(history) % 5 == 0 or force:
#         summary_prompt = f"Summarize this conversation briefly:\n\n{history}"
#         summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()
        
#         try:
#             return summarization_chain.invoke(summary_prompt)
#         except Exception:
#             return history[-500:]  # Fallback
#     else:
#         return history[-500:]  # Use recent 500 chars instead of summarizing

# # Create chat chain
# def create_chat_chain():
#     llm = ChatGroq(model="llama3-8b-8192")
#     prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", message_template)])
#     return prompt | llm | StrOutputParser()

# # Main chatbot loop
# def main():
#     retriever = load_vectorstore()
#     chain = create_chat_chain()
#     conversation_history = []
#     current_stage = "initial"

#     print("Welcome to Dr. PsyRA. Type 'bye doctor' to exit.")

#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() == "bye doctor":
#             print("\nDr. PsyRA: Thank you. Take care!")
#             break

#         context = get_relevant_context(user_input, retriever)
#         summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

#         inputs = {"context": context, "question": user_input, "stage": current_stage, "history": summarized_history}

#         response = chain.invoke(inputs)
        
#         # Debug
#         print(f"Stage:{current_stage}\n")

#         stage_match = re.search(r"\*\*\[Next Stage: (.*?)\]\*\*", response)
#         if stage_match:
#             current_stage = stage_match.group(1).strip().lower()

#         print(f"\nDr. PsyRA: {response}")

#         conversation_history.append(f"User: {user_input}")
#         conversation_history.append(f"Assistant: {response}")

# if __name__ == "__main__":
#     main()
