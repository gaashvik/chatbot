import os
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from langchain.chat_models import init_chat_model
import config

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is required")

os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY

LLM_REACT = init_chat_model("google_genai:gemini-2.5-flash")
LLM_HYPE = init_chat_model("google_genai:gemini-2.5-flash")

mongo_client = MongoClient(MONGODB_URI)
