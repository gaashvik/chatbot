import os
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from langchain_aws import ChatBedrockConverse
from langchain.chat_models import init_chat_model
import config
import boto3

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v2:0"
HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


session = boto3.Session(profile_name="shubhk", region_name="us-east-1")
bedrock_client = session.client(service_name="bedrock-runtime", region_name="us-east-1")


MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is required")

os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY

# os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.getenv("BEDROCK_API")

LLM_HYPE = init_chat_model("google_genai:gemini-2.5-flash")
LLM_SEARCH = init_chat_model("google_genai:gemini-2.5-flash")

mongo_client = MongoClient(MONGODB_URI)
