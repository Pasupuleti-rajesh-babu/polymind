import os
from dotenv import load_dotenv
from typing import Set, Dict

# Determine Project Root dynamically
# Assumes config.py is in polymind_v2 directory, and project root is its parent.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file located at the project root
dotenv_path = os.path.join(PROJECT_ROOT, '.env') # Should resolve to polymind_v2/.env
load_dotenv(dotenv_path)

# --- API Keys & Credentials ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Ensure placeholder is generic
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "polymind")

# --- Model Configurations ---
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Added missing configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Default model for general tasks
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest" # Added for clarity, can be same as GEMINI_MODEL_NAME or specific

# --- Knowledge Graph Schema Constants ---
ALLOWED_RELATIONSHIP_TYPES: Set[str] = {
    "IS_A", "PART_OF", "HAS_PROPERTY", "RELATES_TO", "INFLUENCES", 
    "CAUSES", "EFFECT_OF", "USED_FOR", "LOCATED_IN", "CONTAINS", 
    "EXAMPLE_OF", "SUBCLASS_OF", "SUPERCLASS_OF", "ANALOGOUS_TO", 
    "CONTRASTS_WITH", "MEASURES", "UNITS_OF", "DEFINITION_OF",
    "DERIVED_FROM", "HAS_PART", "MEMBER_OF", "SUBSTANCE_OF",
    # Add more as identified, these are from the old app + common ones
    "RELATED_TO" # Default/Fallback
}

# More detailed canonical map if specific synonyms are common
CANONICAL_REL_MAP: Dict[str, str] = {
    "TYPE_OF": "IS_A",
    "INSTANCE_OF": "IS_A",
    "SUBTYPE_OF": "SUBCLASS_OF",
    "MADE_OF": "SUBSTANCE_OF",
    "HAS_COMPONENT": "HAS_PART",
    "CAN_CAUSE": "CAUSES",
    "LEADS_TO": "CAUSES",
    "RESULTS_IN": "CAUSES",
    "IS_EFFECT_OF": "EFFECT_OF",
    "CAN_BE_USED_FOR": "USED_FOR",
    "IS_IN": "LOCATED_IN",
    "MAY_BE_RELATED_TO": "RELATED_TO",
    "SIMILAR_TO": "ANALOGOUS_TO"
}

# --- Default Domain for Concepts ---
DEFAULT_DOMAIN = "General"
MAX_FEATURES_FOR_PROMPT = 7 # Max number of relations/features to show Gemini per concept in prompts

# --- Logging Configuration (Example) ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- FAISS Configuration ---
# Ensure paths are absolute, based on the project root where 'data' directory should exist
FAISS_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if not os.path.exists(FAISS_DATA_DIR):
    os.makedirs(FAISS_DATA_DIR)
    print(f"Created FAISS data directory: {FAISS_DATA_DIR}")

FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.idx")
FAISS_CONCEPT_LIST_PATH = os.path.join(FAISS_DATA_DIR, "faiss_concepts.json")


if __name__ == "__main__":
    # This block can be used to test if variables are loaded correctly
    print(f"Gemini Key Loaded: {GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE'}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Default Embedding Model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Allowed Relationships: {ALLOWED_RELATIONSHIP_TYPES}") 