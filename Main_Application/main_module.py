from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

# Stages in the conversation
stages = {
    "initial": "Initial Engagement",
    "background": "Background Information",
    "risk": "Risk Assessment",
    "mental_health": "Current Presentation",
    "family": "Family History",
    "planning": "Care Planning"
}

# System Prompt Template
system_template = """
You are Dr. PsyRA, a clinical psychologist specializing in intake interviews.

### Key Responsibilities:
1. Conduct structured assessments
2. Perform risk evaluations
3. Gather background information
4. Maintain warm and empathetic communication

### Guidelines:
- Use ONLY information from the provided context.
- If information isn't in the context, say: "I don't have specific guidelines on that."
- Keep responses **short and to the point** for better understanding.
- Maintain a professional yet warm tone.

### **Conversation Stage: {stage}**
- The current stage determines what information to focus on.
- Ensure responses align with the current stage.

---
**Current conversation history:**  
{history}
"""

# User Message Template
message_template = """
### Context:
{context}

### User Question:
{question}

### Instructions:
1. Answer according to the intake guidelines while maintaining a warm, professional tone.
2. Based on the user's input, determine if the conversation should transition to a new stage.
3. If a stage transition is needed, do it and then continue with your response.

---
"""

# Initialize ChromaDB retriever
def load_vectorstore():
    """Loads ChromaDB retriever for retrieving relevant context."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use the same model as when storing
    vectorstore = Chroma(persist_directory="disorders_chroma_vectoredb", embedding_function=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Retrieve relevant context from ChromaDB
def get_relevant_context(question, retriever):
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

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

# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", message_template)
    ])
    return prompt | llm | StrOutputParser()

# Chatbot loop
def main():
    retriever = load_vectorstore()
    chain = create_chat_chain()
    conversation_history = []
    current_stage = "initial"

    print("Welcome to the Clinical Intake Assistant. Type 'bye doctor' to end the session.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA: Thank you for the session. Take care!")
            break

        # Get relevant context from ChromaDB
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
            
            # Check for stage change format in response (e.g., [Next Stage: background])
            stage_match = re.search(r"\*\*\[Next Stage: (.*?)\]\*\*", response)
            if stage_match:
                new_stage = stage_match.group(1).strip().lower()
                if new_stage in stages:
                    current_stage = new_stage
                    print(f"\n(Stage updated to: {stages[current_stage]})")

            # Remove stage change text from the final response shown to the user
            response = re.sub(r"\*\*\[Next Stage: .*?\]\*\*", "", response).strip()

            print(f"\nDr. PsyRA: {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")

        except Exception as e:
            print("\nDr. PsyRA: I apologize, but I'm having trouble processing that. Could you rephrase your question?")

if __name__ == "__main__":
    main()
