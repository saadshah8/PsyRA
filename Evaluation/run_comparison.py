# run_comparison.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os
import re
import json
import time
from dotenv import load_dotenv

# Import ChatbotEvaluator from the evaluation script
from evaluation import ChatbotEvaluator

# Import the advanced clinical intake chatbot
from main_module import create_chat_chain as create_clinical_chain
from main_module import stages as clinical_stages
from main_module import summarize_conversation as clinical_summarize
from main_module import load_vectorstore, get_relevant_context, save_conversation_data

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Stages in the conversation for baseline bot
stages = {
    "initial": "Initial Engagement",
    "background": "Background Information",
    "risk": "Risk Assessment",
    "mental_health": "Current Presentation",
    "family": "Family History",
    "planning": "Care Planning"
}

# System Prompt Template for baseline bot
system_template = """
You are Dr. PsyRA, a clinical psychologist specializing in intake interviews.

### Key Responsibilities:
1. Conduct structured assessments
2. Perform risk evaluations
3. Gather background information
4. Maintain warm and empathetic communication

### Guidelines:
- Keep responses *short and to the point* for better understanding.
- Maintain a professional yet warm tone.

### *Conversation Stage: {stage}*
- The current stage determines what information to focus on.
- Ensure responses align with the current stage.

---
*Current conversation history:*  
{history}
"""

# User Message Template for baseline bot
message_template = """
### User Question:
{question}

### Instructions:
1. Answer according to the intake guidelines while maintaining a warm, professional tone.
2. Based on the user's input, determine if the conversation should transition to a new stage.
3. If a stage transition is needed, do it and then continue with your response.

---
"""

# Summarize previous conversation history for baseline bot
def summarize_conversation(history):
    if not history:
        return "No previous context"
    
    summary_prompt = f"Summarize the following conversation briefly:\n\n{history}"
    summarization_chain = ChatGroq(model="llama-3.1-8b-instant") | StrOutputParser()
    
    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return history[-500:]  # Fallback: Keep last 500 characters if summarization fails

# Create baseline chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", message_template)
    ])
    return prompt | llm | StrOutputParser()

# Function to run the baseline chatbot
def main():
    """Run the baseline chatbot without vectorstore"""
    baseline_chain = create_chat_chain()
    conversation_history = []
    current_stage = "initial"
    
    # Data collection for evaluation
    chatbot_responses = {
        "Baseline Model": []
    }

    print("Welcome to the Baseline Clinical Intake Assistant. Type 'bye doctor' to end the session.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\nDr. PsyRA (Baseline): Thank you for the session. Take care!")
            # Save conversation data before exiting
            save_conversation_data(chatbot_responses)
            break

        # Summarize conversation history
        summarized_history = summarize_conversation("\n".join(conversation_history[-10:]))

        # Prepare chain inputs
        inputs = {
            "question": user_input,
            "stage": stages[current_stage],
            "history": summarized_history
        }

        # Get response
        try:
            start_time = time.time()
            response = baseline_chain.invoke(inputs)
            response_time = time.time() - start_time
            
            # Check for stage change
            stage_match = re.search(r"\\\[Next Stage: (.?)\]\\*", response)
            if stage_match:
                new_stage = stage_match.group(1).strip().lower()
                if new_stage in stages:
                    current_stage = new_stage
                    print(f"\n(Stage updated to: {stages[current_stage]})")

            # Clean response
            final_response = re.sub(r"\\\[Next Stage: .?\]\\*", "", response).strip()
            print(f"\nDr. PsyRA (Baseline): {final_response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {final_response}")
            
            # Store response data for evaluation
            chatbot_responses["Baseline Model"].append({
                "response": final_response,
                "response_time": response_time,
                "user_input": user_input,
                "stage": current_stage
            })

        except Exception as e:
            print(f"\nDr. PsyRA (Baseline): I apologize, but I'm having trouble processing that. (Error: {e})")

# Interactive chatbot comparison function
def run_comparison():
    """Run interactive comparison between baseline and clinical chatbots, saving conversation for evaluation"""
    # Initialize both chatbots
    baseline_chain = create_chat_chain()
    clinical_chain = create_clinical_chain()
    retriever = load_vectorstore()
    
    # Initialize conversation histories and stages
    baseline_history = []
    clinical_history = []
    baseline_stage = "initial"
    clinical_stage = "initial"
    
    # Initialize response data for evaluation
    conversation_data = {
        "Dr. PsyRA": [],  # Clinical model with vector store
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
        summarized_baseline_history = summarize_conversation("\n".join(baseline_history[-10:]))
        baseline_inputs = {
            "question": user_input,
            "stage": stages[baseline_stage],
            "history": summarized_baseline_history
        }
        
        try:
            start_time = time.time()
            baseline_response = baseline_chain.invoke(baseline_inputs)
            baseline_response_time = time.time() - start_time
            
            # Check for stage change
            stage_match = re.search(r"\\\[Next Stage: (.?)\]\\*", baseline_response)
            if stage_match:
                new_stage = stage_match.group(1).strip().lower()
                if new_stage in stages:
                    baseline_stage = new_stage
                    print(f"(Baseline Stage updated to: {stages[baseline_stage]})")
            
            # Clean response
            baseline_final = re.sub(r"\\\[Next Stage: .?\]\\*", "", baseline_response).strip()
            print(f"Dr. PsyRA (Baseline): {baseline_final}")
            
            # Update history
            baseline_history.append(f"User: {user_input}")
            baseline_history.append(f"Assistant: {baseline_final}")
            
            # Store response for evaluation
            conversation_data["Baseline Model"].append({
                "response": baseline_final,
                "response_time": baseline_response_time,
                "user_input": user_input,
                "stage": baseline_stage
            })
        except Exception as e:
            print(f"Dr. PsyRA (Baseline): I apologize, but I'm having trouble processing that. (Error: {e})")
            conversation_data["Baseline Model"].append({
                "response": "I apologize, but I'm having trouble processing that.",
                "response_time": 0.1,
                "user_input": user_input,
                "stage": baseline_stage
            })
        
        # Process with clinical chatbot
        print("\n--- Dr. PsyRA Model Response ---")
        # Get relevant context
        context = get_relevant_context(user_input, retriever)
        summarized_clinical_history = clinical_summarize("\n".join(clinical_history[-10:]))
        clinical_inputs = {
            "context": context,
            "question": user_input,
            "stage": clinical_stages[clinical_stage],
            "history": summarized_clinical_history
        }
        
        try:
            start_time = time.time()
            clinical_response = clinical_chain.invoke(clinical_inputs)
            clinical_response_time = time.time() - start_time
            
            # Check for stage change
            stage_match = re.search(r"\\\[Next Stage: (.?)\]\\*", clinical_response)
            if stage_match:
                new_stage = stage_match.group(1).strip().lower()
                if new_stage in clinical_stages:
                    clinical_stage = new_stage
                    print(f"(Clinical Stage updated to: {clinical_stages[clinical_stage]})")
            
            # Clean response
            clinical_final = re.sub(r"\\\[Next Stage: .?\]\\*", "", clinical_response).strip()
            print(f"Dr. PsyRA (Clinical): {clinical_final}")
            
            # Update history
            clinical_history.append(f"User: {user_input}")
            clinical_history.append(f"Assistant: {clinical_final}")
            
            # Store response for evaluation
            conversation_data["Dr. PsyRA"].append({
                "response": clinical_final,
                "response_time": clinical_response_time,
                "user_input": user_input,
                "stage": clinical_stage
            })
        except Exception as e:
            print(f"Dr. PsyRA (Clinical): I apologize, but I'm having trouble processing that. (Error: {e})")
            conversation_data["Dr. PsyRA"].append({
                "response": "I apologize, but I'm having trouble processing that.",
                "response_time": 0.1,
                "user_input": user_input,
                "stage": clinical_stage
            })
    
    # Save conversation data
    save_conversation_data(conversation_data, "interactive_comparison_data.json")
    print(f"Conversation data saved to 'interactive_comparison_data.json'")
    
    # Run evaluation
    evaluator = ChatbotEvaluator(["Dr. PsyRA", "Baseline Model"])
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run clinical chatbot or comparison')
    parser.add_argument('--compare', action='store_true', help='Run chatbot comparison in interactive mode')
    args = parser.parse_args()
    
    if args.compare:
        run_comparison()
    else:
        main()  # Run the baseline bot by default