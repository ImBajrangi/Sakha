import os

# Configuration for the CustomerAssistantBot
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY")
DATASET_PATH = "/Users/mr.bajrangi/Code/Company/Projects/CustomerAssistantBot/dataset_refined.json"

# Search settings
MAX_SEARCH_RESULTS = 3

# Personalized response settings
STYLES = {
    "short": "Provide a very concise answer.",
    "long": "Provide a detailed and comprehensive answer.",
    "overview": "Provide a high-level summary of the topic."
}
