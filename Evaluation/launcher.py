#!/usr/bin/env python3
# launcher.py - Main entry point for running the chatbots or comparison
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='Mental Health Chatbot System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --clinical     # Run the clinical chatbot with vectorstore
  python launcher.py --baseline     # Run the baseline chatbot without vectorstore
  python launcher.py --compare      # Run the interactive comparison between both chatbots
  python launcher.py --evaluate     # Run evaluation on saved conversation data
        """
    )
    
    # Create a mutually exclusive group for the main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--clinical', action='store_true', help='Run the clinical chatbot (with vectorstore)')
    group.add_argument('--baseline', action='store_true', help='Run the baseline chatbot (without vectorstore)')
    group.add_argument('--compare', action='store_true', help='Run interactive comparison between both chatbots')
    group.add_argument('--evaluate', action='store_true', help='Run evaluation on existing conversation data')
    
    args = parser.parse_args()
    
    if args.clinical:
        # Run the clinical chatbot
        print("Starting Dr. PsyRA Clinical Chatbot (with vectorstore)...")
        from main_module import main
        main()
    elif args.baseline:
        # Run the baseline chatbot
        print("Starting Baseline Clinical Chatbot (without vectorstore)...")
        from run_comparison import main
        main()
    elif args.compare:
        # Run the interactive comparison
        print("Starting Interactive Comparison Mode...")
        from run_comparison import run_comparison
        run_comparison()
    elif args.evaluate:
        # Run the evaluation on existing data
        print("Running Evaluation on Saved Conversation Data...")
        from evaluation import main
        main()

if __name__ == "__main__":
    main()