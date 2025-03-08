from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from langchain_groq import ChatGroq
from datasets import Dataset
import re

def calculate_ambiguity_score(response):
    """
    Measures response specificity by extracting a numerical score from the LLM response.
    """
    prompt = f"""Analyze this response for ambiguity:
    {response}
    
    Provide only a numeric score between 0 and 1, without explanation.
    """

    llm = ChatGroq(model="llama3-8b-8192")
    raw_response = llm.invoke(prompt).content.strip()

    print(f"Raw LLM Response: {raw_response}")  # Debugging

    # Extract the first number (integer or decimal)
    match = re.search(r"\b\d+(\.\d+)?\b", raw_response)
    
    return float(match.group()) if match else 0.5  # Default to 0.5 if parsing fails


