from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")


system_template = """
You are PsyRA, a highly skilled clinical psychologist conducting structured yet conversational interviews.  
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


def main():
    retriever = load_vectorstore()
    chain = create_chat_chain()
    conversation_history = []
    past_summary = ""  # Initialize past summary
    turn_count = 0  # Track user-assistant exchanges
    
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

        context = get_relevant_context(user_input, retriever)
        conversation_history.append(f"User: {user_input}")

        # Use both past_summary and latest history
        inputs = {"context": context, "question": user_input, "history": past_summary}

        # Get response with timing
        try:
            start_time = time.time()
            response = chain.invoke(inputs)
            response_time = time.time() - start_time
            
            response = clean_response(response)

            print(f"\nPsyRA: {response}")

            conversation_history.append(f"Assistant: {response}")
            turn_count += 1  # Increment turn count
            
            # Store response data for evaluation
            chatbot_responses["PsyRA"].append({
                "response": response,
                "response_time": response_time,
                "user_input": user_input
            })

        except Exception as e:
            print(f"\nPsyRA: I apologize, but I'm having trouble processing that. Could you rephrase your question? (Error: {e})")
            conversation_history.append(f"Assistant: I apologize, but I'm having trouble processing that. Could you rephrase your question?")

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