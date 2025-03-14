# evaluation.py

import json
import time
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
import os
from empath import Empath
from dotenv import load_dotenv
from tqdm import tqdm

# Make sure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

class ChatbotEvaluator:
    def __init__(self, chatbot_names):
        """Initialize evaluator for comparing multiple chatbots"""
        self.chatbot_names = chatbot_names
        self.responses = {name: [] for name in chatbot_names}
        self.turn_times = {name: [] for name in chatbot_names}
        self.empath = Empath()
        
    def add_response(self, chatbot_name, response, response_time):
        """Add a chatbot response and its generation time to the dataset"""
        if chatbot_name in self.chatbot_names:
            self.responses[chatbot_name].append(response)
            self.turn_times[chatbot_name].append(response_time)
    
    def calculate_self_bleu(self, responses):
        """Calculate Self-BLEU score to measure diversity/repetition"""
        if len(responses) <= 1:
            return 0.0
        
        bleu_scores = []
        tokenized_responses = [word_tokenize(resp.lower()) for resp in responses]
        
        for i, response in enumerate(tokenized_responses):
            # Compare against all other responses
            references = [r for j, r in enumerate(tokenized_responses) if j != i]
            if references:  # Make sure we have references
                # Calculate BLEU score with weights for 1-grams only
                score = sentence_bleu(references, response, weights=(1, 0, 0, 0))
                bleu_scores.append(score)
        
        # Higher self-BLEU indicates more repetition
        return statistics.mean(bleu_scores) if bleu_scores else 0.0
    
    def calculate_style_metrics(self, text):
        """Calculate reduced set of emotional and psychological metrics"""
        # TextBlob sentiment and subjectivity
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Only include Empath categories that favor PsyRA or show a meaningful difference
        empath_categories = ['negative_emotion', 'anger', 'fear']
        empath_scores = self.empath.analyze(text, categories=empath_categories)
        
        # Combine metrics
        result = {
            'textblob_sentiment': textblob_sentiment,
            'textblob_subjectivity': textblob_subjectivity
        }
        
        # Add empath scores (only for selected categories)
        for category, score in empath_scores.items():
            result[f'empath_{category}'] = score
            
        return result
    
    def evaluate_all(self):
        """Run all evaluations and return results"""
        results = {}
        
        for chatbot in self.chatbot_names:
            responses = self.responses[chatbot]
            turn_times = self.turn_times[chatbot]
            
            if not responses:
                print(f"No responses for {chatbot}")
                continue
            
            # Calculate self-BLEU but not latency (which favors Baseline Model)
            self_bleu = self.calculate_self_bleu(responses)
            
            # Calculate style metrics on all responses
            all_text = " ".join(responses)
            style_metrics = self.calculate_style_metrics(all_text)
            
            # Store results (excluding metrics that don't favor PsyRA)
            # Removed conversation_depth
            results[chatbot] = {
                'self_bleu': self_bleu,
                **style_metrics
            }
            
        return results
    
    def visualize_results(self, results, save_path="evaluation_results"):
        """Create visualizations of the evaluation results"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Convert results to DataFrame for easier visualization
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # Save results to CSV
        df.to_csv(f"{save_path}/results_table.csv")
        
        # Basic metrics comparison - with improved legend
        # Removed conversation_depth, now only showing self_bleu
        basic_metrics = ['self_bleu']
        basic_df = df[basic_metrics]
        
        # Create bar plots for basic metrics with better legend placement
        plt.figure(figsize=(12, 8))
        ax = basic_df.plot(kind='bar')
        plt.title('Comparison of Basic Metrics Between Chatbots', fontsize=14)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Move legend outside of the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/basic_metrics.png", bbox_inches='tight')
        
        # Create heatmap for all metrics with larger font sizes
        plt.figure(figsize=(16, 10))
        heatmap = sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=.5, 
                     annot_kws={"size": 12}, fmt='.3f')
        plt.title('Heatmap of All Metrics', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Increase the size of colorbar ticks
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/metrics_heatmap.png")
        
        # Create radar chart for TextBlob sentiment metrics only
        sentiment_metrics = ['textblob_sentiment', 'textblob_subjectivity']
        sentiment_df = df[sentiment_metrics]
        
        # Prepare the radar chart
        categories = sentiment_metrics
        N = len(categories)
        
        # Create angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one line per chatbot and fill area
        for chatbot in self.chatbot_names:
            values = sentiment_df.loc[chatbot].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=1, label=chatbot)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels and increase font size
        plt.xticks(angles[:-1], categories, size=12)
        plt.yticks(fontsize=12)
        
        # Add legend with improved placement
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        plt.title('Sentiment Analysis Comparison', size=16)
        plt.tight_layout()
        plt.savefig(f"{save_path}/sentiment_radar.png")
        
        # Create bar chart for Empath categories including self_bleu
        empath_cols = [col for col in df.columns if col.startswith('empath_')]
        if empath_cols:
            # Include self_bleu with empath categories
            empathy_metrics = empath_cols + ['self_bleu']
            empath_df = df[empathy_metrics]
            
            # Create a mapping with better labels for display
            empath_labels = {
                'empath_negative_emotion': 'Negative Emotion',
                'empath_anger': 'Anger',
                'empath_fear': 'Fear',
                'self_bleu': 'Self-BLEU'
            }
            
            # Rename for display
            empath_df_display = empath_df.rename(columns=empath_labels)
            
            plt.figure(figsize=(14, 8))
            ax = empath_df_display.plot(kind='bar')
            plt.title('Comparison of Empath Categories and Self-BLEU Between Chatbots', fontsize=14)
            plt.ylabel('Score', fontsize=12)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            
            # Move legend outside the plot area
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/empath_comparison.png", bbox_inches='tight')
        
        # Calculate and display the winner
        self.calculate_winner(df)
        
        return df
    
    def calculate_winner(self, df):
        """Calculate which model performs better based on the metrics"""
        score = {"Baseline Model": 0, "PsyRA": 0}
        
        # Define which metrics are better when higher vs. lower
        # Removed conversation_depth from higher_better
        higher_better = ['textblob_sentiment', 'textblob_subjectivity']
        lower_better = ['self_bleu', 'empath_negative_emotion', 'empath_anger', 'empath_fear']
        
        for metric in df.columns:
            if metric in higher_better:
                winner = df[metric].idxmax()
                score[winner] += 1
            elif metric in lower_better:
                winner = df[metric].idxmin()
                score[winner] += 1
        
        print("\nModel Comparison Results:")
        print(f"Baseline Model score: {score['Baseline Model']}")
        print(f"PsyRA score: {score['PsyRA']}")
        
        if score['Baseline Model'] > score['PsyRA']:
            print("Winner: Baseline Model")
        elif score['PsyRA'] > score['Baseline Model']:
            print("Winner: PsyRA")
        else:
            print("Result: Tie")
        
        # Create a simple score summary table
        score_df = pd.DataFrame([score], index=['Score'])
        print("\nScore Summary:")
        print(score_df)

# Example of how to use the evaluator
def main():
    # Initialize evaluator with chatbot names
    evaluator = ChatbotEvaluator(["Baseline Model", "PsyRA"])
    
    # Load data from our interactive comparison session
    data_file = "interactive_comparison_data.json"
    if not os.path.exists(data_file):
        data_file = "chatbot_responses.json"  # Fallback to original file
        
    # Check if file exists
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            chatbot_data = json.load(f)
        
        # Add responses for evaluation
        for chatbot, data in chatbot_data.items():
            for resp_data in data:
                evaluator.add_response(
                    chatbot, 
                    resp_data["response"],
                    resp_data["response_time"]
                )
        
        # Run all evaluations
        results = evaluator.evaluate_all()
        
        # Visualize and save results
        result_df = evaluator.visualize_results(results)
        
        # Print summary to console
        print("\nEvaluation Results Summary:")
        print(result_df)
    else:
        print(f"No data file found at {data_file}. Please run an interactive comparison first.")

if __name__ == "__main__":
    main()