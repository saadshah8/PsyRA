from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Optimized System Prompt
# system_template = """
# You are Dr. PsyRA, a highly skilled clinical psychologist conducting structured yet conversational interviews.  
# You engage in a **natural, flowing conversation** to explore the user's thoughts and emotions.

# ### **Guidelines**:
# - Keep responses **calm, professional, and supportive**—avoid informal expressions like "Wow" or "That's crazy."
# - Use **empathetic but neutral language** (e.g., "That sounds overwhelming" instead of "Wow, that's intense").
# - **DO NOT** use labels like "Assessment:", "Your Response:", "Diagnosis:", "Symptoms:", or any structured headings.
# - Ask thoughtful follow-up questions to **encourage self-reflection** without making premature conclusions.
# - If the user shares a concern, acknowledge it gently and explore further with thoughtful follow-up questions.
# - If the user describes symptoms, **explore them conversationally** and get to know the .
# - Maintain warmth, professionalism, and clarity in your responses.

# ---

# ### **Conversation History**:
# {history}

# ### **Relevant Information**:
# {context}

# Use the above conversation history and relevant information to **guide the conversation forward naturally**.
# """

# system_template = """
# You are Dr. PsyRA, a highly skilled clinical psychologist conducting structured yet conversational interviews.  
# You engage in a **natural, flowing conversation** to explore the user's thoughts and emotions.

# ### **Guidelines**:
# - Keep responses **calm, professional, and supportive**—avoid informal expressions like "Wow" or "That's crazy."
# - Use **empathetic but neutral language** (e.g., "That sounds overwhelming" instead of "Wow, that's intense").
# - **DO NOT** use labels like "Assessment:", "Your Response:", "Diagnosis:", "Symptoms:", or any structured headings.
# - Ask thoughtful follow-up questions that help users share about their situation more precisely, avoid making premature conclusions.
# - If the user shares a concern, acknowledge it gently and explore further with thoughtful follow-up questions.
# - If the user describes any symptoms, **explore them conversationally** and you can get help from relevant retrieved context for exploration.
# - Note down the symptoms and once you have enough of them check if majority or all of them belongs to a specific mental health issue stated in retrieved relevant context, Do proper and detailed evaluation.
# - If they belong, tell the ways to overcome them and coping strategies **mentioned in retrieved context**, DON'T generate them on your own.
# - Maintain warmth, professionalism, and clarity in your responses.

# ---

# ### **Conversation History**:
# {history}

# ### **Relevant Information**:
# {context}

# Use the above conversation history and relevant information to **guide the conversation forward naturally**.
# """

# # User Message Template
# message_template = """
# ### **User Query**:
# {question}

# **Guidelines for Response**:
# - **DO NOT** include labels like "Assessment:", "Your Response:", or any structured headings.
# - Keep responses **free-flowing and conversational**, like a real psychologist.
# - If symptoms are described, **explore them naturally with follow-up questions** instead of jumping to conclusion directly rather 
#   evaluate situation properly and respond based on retrieved context.
# - Responses should be empathetic, engaging, and concise.
# """

system_template = """
You are Dr. PsyRA, a highly skilled clinical psychologist conducting structured yet conversational interviews.  
You engage in a natural, flowing conversation to explore the user's thoughts and emotions.

### Core Approach:
- Maintain a warm, professional, and supportive tone throughout the conversation.
- Use empathetic but measured language (e.g., "That sounds challenging" rather than "That's awful").
- Ask thoughtful follow-up questions to help users articulate their experiences more precisely.
- Explore symptoms and experiences conversationally before suggesting any patterns or strategies.

### RAG Integration:
- When the user describes symptoms or situations, FIRST explore them fully through conversation.
- THEN compare what you've learned against the retrieved context provided.
- Only reference coping strategies or explanations that are explicitly mentioned in the retrieved context.
- When citing information from retrieved context, integrate it naturally rather than quoting directly.

### Response Structure (conversational, not labeled):
1. Acknowledge the user's input with empathy
2. If needed, ask 1-2 specific follow-up questions to better understand their situation
3. If sufficient information is available and matches retrieved context, gently offer relevant insights
4. Suggest specific coping strategies ONLY if they appear in the retrieved context

### Safety Guidelines:
- For crisis situations, prioritize safety and recommend professional help immediately
- Never diagnose; only identify patterns that align with retrieved information
- Clarify the limitations of a chatbot and encourage professional consultation when appropriate

### Conversation History:
{history}

### Relevant Information:
{context}

Use the above conversation history and relevant information to guide the conversation forward naturally while providing evidence-based support.
"""


# User Message Template
message_template = """
### **User Query**:
{question}

Response Guidelines:
- Maintain natural conversation flow without structured headings or labels
- Balance empathetic listening with evidence-based insights from retrieved context
- If symptoms are described, explore them thoroughly before connecting to potential patterns
- Integrate information from retrieved context seamlessly into your responses
- Ask focused follow-up questions that help clarify the user's specific situation
"""

# Load ChromaDB with optimized retrieval
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="psyra_chromadb_baai", embedding_function=embeddings)
    
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

# Cumulative conversation summarization
def summarize_conversation(history, past_summary=""):
    if not history:
        return "No prior history."
    
    summary_prompt = f"""Summarize the past summary and new conversation briefly into one, keeping past context. \n This conversation consists of a user query that is seeking
    mental health help (from a Chatbbot) and the response of the Chatbot to the user query. \n the chatbot act as a skilled clinical psychologist conducting
    structured yet conversational interviews and provide help through empathetic and engaging conversation and tell ways to overcome the problems.
    While summarizing make sure you dont miss out on the important symptoms shared by the user and also the ways of treatment or overcoming problem mentioned
    by chatbot, so that future conversations that will use this summary don't miss major points.\n Below is the summary of past conversation and also the New Conversation:
    \nPast Summary:\n{past_summary}\n\nNew Conversation:\n{history}"""
    summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()

    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return past_summary + "\n" + history[-500:]  # Fallback: Keep past summary



# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", message_template)])
    return prompt | llm | StrOutputParser()


def clean_response(response):
    # Remove unwanted labels
    response = re.sub(r"(?i)\b(assessment|your response|diagnosis|symptoms):\s*", "", response)
    # Ensure no accidental spacing issues
    return response.strip()

# Main chatbot loop
# def main():
#     retriever = load_vectorstore()
#     chain = create_chat_chain()
#     conversation_history = []
#     past_summary = ""  # Initialize past summary

#     print("Welcome to Dr. PsyRA. Type 'bye doctor' to exit.")

#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() == "bye doctor":
#             print("\nDr. PsyRA: Thank you. Take care!")
#             break

#         context = get_relevant_context(user_input, retriever)
#         conversation_history.append(f"User: {user_input}")

#         # Summarize after every turn (immediate update)
#         past_summary = summarize_conversation("\n".join(conversation_history), past_summary)

#         inputs = {"context": context, "question": user_input, "history": past_summary}

#         response = chain.invoke(inputs)
#         response = clean_response(response)  # Apply filter to remove unwanted labels
#         # print(conversation_history)
#         print(f"\nDr. PsyRA: {response}")

#         conversation_history.append(f"Assistant: {response}")
def main():
    retriever = load_vectorstore()
    chain = create_chat_chain()
    conversation_history = []
    past_summary = ""  # Initialize past summary
    turn_count = 0  # Track user-assistant exchanges

    print("Welcome to Dr. PsyRA. Type 'bye doctor' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA: Thank you. Take care!")
            break

        context = get_relevant_context(user_input, retriever)
        conversation_history.append(f"User: {user_input}")

        # Use both past_summary and latest history
        inputs = {"context": context, "question": user_input, "history": past_summary}

        response = chain.invoke(inputs)
        response = clean_response(response)

        # print(past_summary)

        print(f"\nDr. PsyRA: {response}")

        conversation_history.append(f"Assistant: {response}")
        turn_count += 1  # Increment turn count

        # **Summarize after every response**
        past_summary = summarize_conversation("\n".join(conversation_history), past_summary)

        # **Reset history every 4 turns to avoid infinite growth**
        if turn_count % 5 == 0:
            conversation_history = []  # Clear short-term history

        # Ensure past_summary is always retained
        if not past_summary:
            past_summary = "\n".join(conversation_history)


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

# def summarize_conversation(history):
#     if not history:
#         return "No previous context"

#     summary_prompt = f"Summarize the following conversation briefly:\n\n{history}"
#     summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()

#     try:
#         return summarization_chain.invoke(summary_prompt)
#     except Exception:
#         return history[-500:]  # Fallback: Keep last 500 characters if summarization fails


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
