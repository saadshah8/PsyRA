# run_comparison.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os
import json
import time
import argparse
from dotenv import load_dotenv

# Import ChatbotEvaluator from the evaluation script
from evaluation import ChatbotEvaluator

# Import the updated clinical intake chatbot functionality
from upd_main import (
    create_chat_chain as create_clinical_chain,
    summarize_conversation as clinical_summarize,
    load_vectorstore, 
    get_relevant_context, 
    save_conversation_data
)

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# System Prompt Template for baseline bot - using the same template as the clinical chatbot
system_template = """
You are PsyRA, a highly skilled clinical psychologist specializing in structured intake interviews.

### **Guidelines**:
- Keep responses **concise** and professional, with warmth.

### **History:** {history}
"""

# User Message Template for baseline bot - similar to clinical but without the context
message_template = """
### **User Query**:
{question}

**Instructions**:
1. Respond based on the intake guidelines.
"""

# Create baseline chat chain - same structure as clinical but without RAG
def create_baseline_chain():
    llm = ChatGroq(model="llama3-8b-8192")  # Using the same model as clinical
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", message_template)])
    return prompt | llm | StrOutputParser()

# Function to run the baseline chatbot
def run_baseline():
    """Run the baseline chatbot without vectorstore"""
    baseline_chain = create_baseline_chain()
    conversation_history = []
    
    # Data collection for evaluation
    chatbot_responses = {
        "Baseline Model": []
    }

    print("Welcome to the Baseline Clinical Intake Assistant. Type 'bye doctor' to end the session.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nPsyRA (Baseline): Thank you for the session. Take care!")
            # Save conversation data before exiting
            save_conversation_data(chatbot_responses)
            break

        # Use the same summarization technique as the clinical chatbot
        summarized_history = clinical_summarize("\n".join(conversation_history[-10:]))

        # Prepare chain inputs
        inputs = {
            "question": user_input,
            "history": summarized_history
        }

        # Get response
        try:
            start_time = time.time()
            response = baseline_chain.invoke(inputs)
            response_time = time.time() - start_time
            
            print(f"\nPsyRA (Baseline): {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
            # Store response data for evaluation
            chatbot_responses["Baseline Model"].append({
                "response": response,
                "response_time": response_time,
                "user_input": user_input
            })

        except Exception as e:
            print(f"\nPsyRA (Baseline): I apologize, but I'm having trouble processing that. (Error: {e})")

# Interactive chatbot comparison function
def run_comparison():
    """Run interactive comparison between baseline and clinical chatbots, saving conversation for evaluation"""
    # Initialize both chatbots using imported functions
    baseline_chain = create_baseline_chain()
    clinical_chain = create_clinical_chain()
    retriever = load_vectorstore()
    
    # Initialize conversation histories
    baseline_history = []
    clinical_history = []
    
    # Initialize response data for evaluation
    conversation_data = {
        "PsyRA": [],  # Clinical model with vector store
        "Baseline Model": []  # Basic model without vector store
    }
    
    print("=" * 80)
    print("INTERACTIVE CHATBOT COMPARISON")
    print("=" * 80)
    print("You'll be conversing with both chatbots simultaneously.")
    print("Each message you send will be processed by both models.")
    print("Type 'bye doctor' to end the session and run evaluation.")
    print("=" * 80)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nSession ended. Running evaluation...")
            break
        
        # Process with baseline chatbot
        print("\n--- Baseline Model Response ---")
        summarized_baseline_history = clinical_summarize("\n".join(baseline_history[-10:]))
        baseline_inputs = {
            "question": user_input,
            "history": summarized_baseline_history
        }
        
        try:
            start_time = time.time()
            baseline_response = baseline_chain.invoke(baseline_inputs)
            baseline_response_time = time.time() - start_time
            
            print(f"PsyRA (Baseline): {baseline_response}")
            
            # Update history
            baseline_history.append(f"User: {user_input}")
            baseline_history.append(f"Assistant: {baseline_response}")
            
            # Store response for evaluation
            conversation_data["Baseline Model"].append({
                "response": baseline_response,
                "response_time": baseline_response_time,
                "user_input": user_input
            })
        except Exception as e:
            print(f"PsyRA (Baseline): I apologize, but I'm having trouble processing that. (Error: {e})")
            conversation_data["Baseline Model"].append({
                "response": "I apologize, but I'm having trouble processing that.",
                "response_time": 0.1,
                "user_input": user_input
            })
        
        # Process with clinical chatbot
        print("\n--- PsyRA Model Response ---")
        # Get relevant context
        context = get_relevant_context(user_input, retriever)
        summarized_clinical_history = clinical_summarize("\n".join(clinical_history[-10:]))
        clinical_inputs = {
            "context": context,
            "question": user_input,
            "history": summarized_clinical_history
        }
        
        try:
            start_time = time.time()
            clinical_response = clinical_chain.invoke(clinical_inputs)
            clinical_response_time = time.time() - start_time
            
            print(f"PsyRA (Clinical): {clinical_response}")
            
            # Update history
            clinical_history.append(f"User: {user_input}")
            clinical_history.append(f"Assistant: {clinical_response}")
            
            # Store response for evaluation
            conversation_data["PsyRA"].append({
                "response": clinical_response,
                "response_time": clinical_response_time,
                "user_input": user_input
            })
        except Exception as e:
            print(f"PsyRA (Clinical): I apologize, but I'm having trouble processing that. (Error: {e})")
            conversation_data["PsyRA"].append({
                "response": "I apologize, but I'm having trouble processing that.",
                "response_time": 0.1,
                "user_input": user_input
            })
    
    # Save conversation data
    save_conversation_data(conversation_data, "interactive_comparison_data.json")
    print(f"Conversation data saved to 'interactive_comparison_data.json'")
    
    # Run evaluation
    evaluator = ChatbotEvaluator(["PsyRA", "Baseline Model"])
    
    # Add responses to evaluator
    for chatbot, responses in conversation_data.items():
        for resp_data in responses:
            evaluator.add_response(
                chatbot,
                resp_data["response"],
                resp_data["response_time"]
            )
    
    # Run evaluation
    print("\nEvaluating responses...")
    results = evaluator.evaluate_all()
    
    # Visualize results
    print("\nGenerating visualizations...")
    result_df = evaluator.visualize_results(results, save_path="interactive_evaluation_results")
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print(result_df)
    
    print("\nDetailed results and visualizations saved to 'interactive_evaluation_results' directory")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clinical chatbot or comparison')
    parser.add_argument('--compare', action='store_true', help='Run chatbot comparison in interactive mode')
    args = parser.parse_args()
    
    if args.compare:
        run_comparison()
    else:
        run_baseline()  # Run the baseline bot by default