from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from quality_metrics import calculate_ambiguity_score
import re
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from functools import wraps
import psutil
import pandas as pd
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from plots import generate_research_visualizations
from groq import Groq

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

# Initialize Groq client
client = Groq(api_key=groq_api)  # Replace with your actual API key

# Global dictionary to store performance metrics
performance_metrics = {}

# Function to estimate token count (Basic Approximation)
def estimate_token_count(text):
    return len(text.split())  # Approximate by counting words


# Performance Monitor Decorator with improved memory tracking
def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before measurement to get more consistent results
        import gc
        gc.collect()
        
        # Measure initial state
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure final state
        end_time = time.time()
        end_mem = process.memory_info().rss
        elapsed = end_time - start_time
        mem_used = (end_mem - start_mem) / 1024**2  # Convert bytes to MB
        
        # Better token estimation
        token_count = 0
        if isinstance(result, str):
            # More accurate token estimation
            token_count = len(result.split()) * 1.3  # Assume ~1.3 tokens per word
        
        # Ensure we don't record negative memory (which is likely measurement error)
        mem_used = max(0, mem_used)
        
        # Store metrics in the global dictionary
        if func.__name__ not in performance_metrics:
            performance_metrics[func.__name__] = []

        performance_metrics[func.__name__].append({
            "time": elapsed,
            "memory": mem_used,
            "tokens": token_count,
            "timestamp": time.time()  # Add timestamp for temporal analysis
        })

        return result

    return wrapper

# Add this function to display a more detailed performance summary at the end
def display_performance_summary():
    print("\n===== DETAILED PERFORMANCE SUMMARY =====")
    
    # Calculate overall stats
    total_time = 0
    total_memory = 0
    total_tokens = 0
    func_summaries = {}
    
    for func_name, metrics in performance_metrics.items():
        if not metrics:
            continue
            
        func_time = sum(m["time"] for m in metrics)
        func_memory = max([m["memory"] for m in metrics], default=0)  # Use max memory as indicator
        func_tokens = sum(m["tokens"] for m in metrics)
        
        total_time += func_time
        total_memory += func_memory
        total_tokens += func_tokens
        
        # Calculate averages
        avg_time = func_time / len(metrics)
        avg_tokens = func_tokens / len(metrics) if func_tokens > 0 else 0
        
        func_summaries[func_name] = {
            "calls": len(metrics),
            "total_time": func_time,
            "avg_time": avg_time,
            "max_memory": func_memory,
            "total_tokens": func_tokens,
            "avg_tokens": avg_tokens,
        }
    
    # Print overall stats
    print(f"Total conversation time: {total_time:.2f} seconds")
    print(f"Maximum memory usage: {total_memory:.2f} MB")
    print(f"Total tokens processed/generated: {int(total_tokens)}")
    
    # Print per-function breakdown
    print("\nFunction-level breakdown:")
    for func_name, summary in func_summaries.items():
        print(f"\n{func_name}:")
        print(f"  Calls: {summary['calls']}")
        print(f"  Total time: {summary['total_time']:.4f} seconds ({(summary['total_time']/total_time*100):.1f}% of total)")
        print(f"  Avg time per call: {summary['avg_time']:.4f} seconds")
        print(f"  Max memory usage: {summary['max_memory']:.2f} MB")
        print(f"  Total tokens: {int(summary['total_tokens'])}")
        print(f"  Avg tokens per call: {int(summary['avg_tokens'])}")
    
    # Calculate bottleneck information
    slowest_func = max(func_summaries.items(), key=lambda x: x[1]['total_time'])
    highest_memory = max(func_summaries.items(), key=lambda x: x[1]['max_memory'])
    
    print("\nPerformance bottlenecks:")
    print(f"  Slowest function: {slowest_func[0]} ({slowest_func[1]['total_time']:.2f} seconds total)")
    print(f"  Highest memory usage: {highest_memory[0]} ({highest_memory[1]['max_memory']:.2f} MB)")


# Decorate critical functions
@performance_monitor
def get_relevant_context(question, retriever):
    """Retrieve relevant context from the database."""
    docs = retriever.invoke(question)
    # Convert retrieved documents to string to ensure compatibility
    if isinstance(docs, list):
        return "\n".join([str(doc.page_content) if hasattr(doc, 'page_content') else str(doc) for doc in docs])
    return str(docs)

@performance_monitor
def summarize_conversation(history):
    """Summarize conversation history."""
    if not history:
        return "No previous context"

    summary_prompt = f"Summarize the following conversation briefly:\n\n{history}"
    summarization_chain = ChatGroq(model="llama3-8b-8192") | StrOutputParser()

    try:
        return summarization_chain.invoke(summary_prompt)
    except Exception:
        return history[-500:]  # Fallback: Keep last 500 characters if summarization fails

@performance_monitor
def chain_invoke(chain, inputs):
    """Invoke the model chain."""
    return chain.invoke(inputs)

# Debugging Step: Check if metrics are being collected
def debug_metrics():
    """Debugging function to check if metrics are being collected."""
    print("Debugging Metrics Collection:")
    for func in [get_relevant_context, summarize_conversation, chain_invoke]:
        if func.__name__ in performance_metrics:
            print(f"{func.__name__}: Metrics found - {len(performance_metrics[func.__name__])} entries")
        else:
            print(f"{func.__name__}: No metrics found (ERROR)")
    print("\n")

# Function to collect performance data
def collect_performance_data():
    data = []
    for func_name, metrics in performance_metrics.items():
        for metric in metrics:
            data.append({
                "stage": func_name,
                "latency": metric["time"],
                "tokens": metric["tokens"],
                "memory": metric["memory"]
            })
    return pd.DataFrame(data)

# Function to perform ANOVA and Tukey HSD
def perform_anova_tukey(data):
    # Check if there are at least two groups with sufficient data
    if len(data["stage"].unique()) < 2:
        print("Not enough groups for ANOVA and Tukey HSD. Skipping tests.")
        return

    try:
        # Perform ANOVA
        anova_result = f_oneway(
            *[data[data["stage"] == stage]["latency"] for stage in data["stage"].unique()]
        )
        print(f"ANOVA Result: {anova_result}")
    except Exception as e:
        print(f"Error performing ANOVA: {e}")

    try:
        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(
            endog=data["latency"],
            groups=data["stage"],
            alpha=0.05
        )
        print(tukey_result)
    except Exception as e:
        print(f"Error performing Tukey HSD: {e}")

# Initialize ChromaDB retriever
def load_vectorstore():
    """Loads ChromaDB retriever for retrieving relevant context."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use the same model as when storing
    vectorstore = Chroma(persist_directory="disorders_chroma_vectoredb", embedding_function=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create chat chain
def create_chat_chain():
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", message_template)
    ])
    return prompt | llm | StrOutputParser()

# Fixed Evaluate retrieval consistency function
def evaluate_retrieval_consistency(retriever, num_trials=5):
    """
    Tests retrieval stability across multiple trials
    """
    test_questions = [
        "What are the symptoms of depression?",
        "How to manage anxiety attacks?",
        "Best therapies for PTSD?"
    ]
    
    results = {}
    
    for q in test_questions:
        contexts = []
        for _ in range(num_trials):
            # Get context and ensure it's a string
            context = get_relevant_context(q, retriever)
            contexts.append(context)
        
        # Calculate similarity between retrievals
        similarity_scores = []
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        for i in range(num_trials-1):
            vec1 = embeddings.embed_query(contexts[i])
            vec2 = embeddings.embed_query(contexts[i+1])
            similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            similarity_scores.append(similarity)
        
        results[q] = {
            'avg_similarity': np.mean(similarity_scores),
            'std_dev': np.std(similarity_scores)
        }
    
    return results

# Chatbot loop
def main():
    # Reset performance metrics at start
    global performance_metrics
    performance_metrics = {}
    
    retriever = load_vectorstore()
    
    # Debugging Step: Check metrics collection before starting
    debug_metrics()
    
    # Evaluate retrieval consistency
    retrieval_metrics = evaluate_retrieval_consistency(retriever)
    print(f"Retrieval Consistency Metrics: {retrieval_metrics}")
    
    chain = create_chat_chain()
    conversation_history = []
    current_stage = "initial"

    print("Welcome to the Clinical Intake Assistant. Type 'bye doctor' to end the session.")

    while True:
        user_input = input("\\nYou: ").strip()
        if user_input.lower() == "bye doctor":
            print("\\nDr. PsyRA: Thank you for the session. Take care!")
            
            # Generate and display research-quality visualizations
            print("\\nGenerating research-quality performance visualizations...")
            if len(performance_metrics) >= 1:
                # Call our advanced visualization function
                generate_research_visualizations(performance_metrics)
                # Display summary of traditional performance metrics for comparison
                display_performance_summary()
            else:
                print("Not enough data collected for analysis.")
            break

        # Get relevant context from ChromaDB
        context = get_relevant_context(user_input, retriever)

        # Summarize conversation history
        history_text = "\\n".join(conversation_history[-10:]) if conversation_history else ""
        summarized_history = summarize_conversation(history_text)

        # Prepare chain inputs
        inputs = {
            "context": context,
            "question": user_input,
            "stage": stages[current_stage],
            "history": summarized_history
        }

        # Get response
        try:
            response = chain_invoke(chain, inputs)
            
            # Check for stage change format in response (e.g., [Next Stage: background])
            stage_match = re.search(r"\\*\\*\\[Next Stage: (.*?)\\]\\*\\*", response)
            if stage_match:
                new_stage = stage_match.group(1).strip().lower()
                if new_stage in stages:
                    current_stage = new_stage
                    # Add stage information to performance metrics for better visualization
                    for func_name in performance_metrics:
                        if performance_metrics[func_name] and len(performance_metrics[func_name]) > 0:
                            # Add stage info to the most recent performance entry
                            performance_metrics[func_name][-1]['stage'] = new_stage
                    
                    print(f"\\n(Stage updated to: {stages[current_stage]})")

            response = re.sub(r"\\*\\*\\[Next Stage: .*?\\]\\*\\*", "", response).strip()
            print(f"\\nDr. PsyRA: {response}")
            
            # Evaluate response quality
            ambiguity = None
            try:
                ambiguity = calculate_ambiguity_score(response)
                print(f"Ambiguity Score: {ambiguity}")  # Debugging Output
            except Exception as e:
                print(f"Error calculating ambiguity: {e}")

            # Add response with metrics to conversation history
            conversation_history.append(f"User: {user_input}\\nDr. PsyRA: {response}\\nAmbiguity Score: {ambiguity}")

            # Print only brief performance metrics after each exchange
            # Only show the most important metrics to keep the output clean
            print("\\nLatest Performance:")
            for func_name, metrics in performance_metrics.items():
                if metrics:  # Check if we have metrics for this function
                    latest_metrics = metrics[-1]  # Get the most recent metrics
                    print(f"{func_name}: {latest_metrics['time']:.2f}s, {int(latest_metrics['tokens'])} tokens")

        except Exception as e:
            print(f"\\nError: {e}")


if __name__ == "__main__":
    main()