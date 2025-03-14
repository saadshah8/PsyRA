# from llama_index.core.tools import FunctionTool
# from llama_index.core.agent import ReActAgent
# from llama_index.core.agent import AgentRunner
# from llama_index.llms.groq import Groq
# from pymongo import MongoClient
# from llama_index.llms.openai import OpenAI




# # MongoDB Configuration
# MONGO_URI = "mongodb://localhost:27017"
# DATABASE_NAME = "morpheus"
# COLLECTION_NAME = "campaigns"

# client = MongoClient(MONGO_URI)
# db = client[DATABASE_NAME]
# collection = db[COLLECTION_NAME]

# def get_campaigns_by_date(start_date: str, end_date: str):
#    # """Fetch campaigns data within a given date range."""
#     query = {
#         "period.startDate": {"$gte": start_date},
#         "period.endDate": {"$lte": end_date}
#     }
#     campaigns = list(collection.find(query, {"_id": 0}))  # Exclude _id from results
#     return campaigns if campaigns else [{"message": "No campaigns found in this date range"}]


# # Define the function
# def fetch_campaigns(start_date: str, end_date: str):
#    # """Fetch campaigns data based on user-specified date range."""
#     return get_campaigns_by_date(start_date, end_date)


# # Your LLM
# api_key = "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"
# llm = Groq(model="llama3-70b-8192" , api_key = api_key)


# # system_prompt = "You are an AI assistant that retrieves campaign data from MongoDB based on user queries."
# fetch_campaigns_tool = FunctionTool.from_defaults(fetch_campaigns)


# # Convert function into a tool
# fetch_campaigns_tool = FunctionTool.from_defaults(fetch_campaigns)
# agent = ReActAgent.from_tools([fetch_campaigns_tool], llm=llm, verbose=False, max_iterations=4)
# response = agent.chat("I want the data of campaign reports from May 2024 to July 2025")
# print(response)
