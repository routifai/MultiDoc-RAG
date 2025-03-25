# file: rag_chatbot/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def get_api_key() -> str:
    """Retrieve API key from environment or raise error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

@dataclass
class ChatConfig:
    """Configuration for the chat application."""
    api_key: str = get_api_key()
    model: str = "gpt-4o-mini"
    exit_commands: set[str] = frozenset({"/exit", "exit", "quit", "/quit"})

    def __init__(self):
        raise TypeError("ChatConfig is not meant to be instantiated")

# Updated model
EMBEDDING_MODEL = "text-embedding-ada-002"  # upgraded to a stronger embedding model

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Increased the number of chunks to retrieve for better context coverage
NUM_CHUNKS_TO_RETRIEVE = 5
SIMILARITY_METRIC = "cosine"

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

MEMORY_SIZE = 10
MAX_CONTEXT_LENGTH = 5000
MAX_TOTAL_TOKENS = 50000
MAX_SUB_QUERIES = 3
