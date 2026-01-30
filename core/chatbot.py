import json
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
import config
from .searcher import WebSearcher
from .synthesizer import LocalSynthesizer

class FastChatbot:
    def __init__(self, dataset_path=config.DATASET_PATH):
        if not os.path.exists(dataset_path):
             # Fallback to current directory for portability
             dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")
             
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        self.dataset_path = dataset_path
        self.dataset = data.get('dataset', [])
        self.searcher = WebSearcher()
        self.synthesizer = LocalSynthesizer()
        self.last_query = ""
        self.last_response = ""
        
        # Initialize DL Semantic Model
        print("Loading DL Semantic Model (Sentence-BERT)...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset_embeddings = None
        self._update_embeddings()

    def _update_embeddings(self):
        """Update vector embeddings for the current dataset."""
        instructions = [item['instruction'] for item in self.dataset]
        self.dataset_embeddings = self.semantic_model.encode(instructions, convert_to_tensor=True)
        print(f"DL Semantic weights updated: {len(instructions)} entries.")

    def save_learning(self, query, correct_response):
        """Append new user-corrected data to the dataset file."""
        new_entry = {"instruction": query, "response": correct_response}
        self.dataset.append(new_entry)
        
        try:
            with open(self.dataset_path, 'w') as f:
                json.dump({"dataset": self.dataset}, f, indent=4)
            self._update_embeddings() # Re-index for DL search
            return True
        except Exception as e:
            print(f"Failed to save learning: {e}")
            return False

    def clean_text(self, text):
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def parse_style(self, user_query):
        # Check for style prefix like "short: " or "overview: "
        for style in config.STYLES:
            if user_query.lower().startswith(f"{style}:"):
                return style, user_query[len(style)+1:].strip()
        return None, user_query

    def get_response(self, user_query):
        style, clean_query = self.parse_style(user_query)
        self.last_query = clean_query # Store for potential correction
        
        # 1. DL Semantic Match
        query_embedding = self.semantic_model.encode(clean_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.dataset_embeddings)[0]
        
        # Get the highest scoring match
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
        
        # Threshold: 0.65 for high-confidence semantic match
        if best_score > 0.65:
            best_match = self.dataset[best_idx]['response']
            print(f"(DL Semantic Match Found: score {best_score:.2f})")
            
            if best_score > 0.85 or not style:
                return best_match
            else:
                return self.synthesizer.synthesize(clean_query, best_match, style)

        # 2. Web Search Fallback
        print(f"(Weak/No match, falling back to web search for: '{clean_query}')...")
        context = self.searcher.search(clean_query)
        
        if context:
            print(f"Web search found context ({len(context)} chars). Synthesizing...")
            response = self.synthesizer.synthesize(clean_query, context, style)
            self.last_response = response # Store for /good reinforcement
            if response and len(response.strip()) > 5:
                return response
        
        return "Radhe Radhe! I'm here to help, but I couldn't find a specific answer for that online currently. Please try asking specifically about our Sattvic food, Chitra Vrinda art, or Vrinda Tours guides!"
