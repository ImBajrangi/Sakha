import os

# Configuration for the Sakha AI Bot
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY") # Deprecated
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_refined.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_output")
USE_LOCAL_MODEL = True

# Search settings
MAX_SEARCH_RESULTS = 3

# Personalized response settings
STYLES = {
    "short": "Provide a very concise answer.",
    "long": "Provide a detailed and comprehensive answer.",
    "overview": "Provide a high-level summary of the topic."
}
