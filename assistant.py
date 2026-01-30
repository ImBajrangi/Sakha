import json
import os
import re
import torch
import logging
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
import config

class WebSearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=config.MAX_SEARCH_RESULTS):
        # Clean query: Remove common punctuation that breaks DDGS
        clean_q = re.sub(r'[^\w\s]', '', query).strip()
        if not clean_q: return ""
        
        # Domain expansion
        if "vrindavan" not in clean_q.lower() and "mathura" not in clean_q.lower() and len(clean_q.split()) < 4:
            clean_q += " in Vrindavan"

        import time
        for attempt in range(2): # Simple retry for ratelimits
            try:
                results = []
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(clean_q, max_results=max_results, region='in-en')]
                
                if not results:
                    with DDGS() as ddgs:
                        results = [r for r in ddgs.news(clean_q, max_results=max_results, region='in-en')]
                
                if results:
                    context = "\n".join([f"Source: {r.get('title','')}\nSnippet: {r.get('body','') or r.get('snippet','')}" for r in results if r.get('body') or r.get('title')])
                    return context
            except Exception as e:
                if "402" in str(e) or "Ratelimit" in str(e):
                    print(f"Search ratelimit hit. Retrying in 2s...")
                    time.sleep(2)
                    continue
                print(f"Search failure for '{clean_q}': {e}")
                break
        return ""

from transformers import pipeline

# Suppress some transformers warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class LocalSynthesizer:
    def __init__(self, model_name=config.MODEL_PATH):
        print(f"Loading local model from {model_name}...")
        
        # Use MPS (Mac GPU) if available
        device = -1
        if torch.backends.mps.is_available():
            device = "mps"
            print("Device set to use mps:0")
        elif torch.cuda.is_available():
            device = 0
            
        self.generator = pipeline("text-generation", model=model_name, device=device)
        print("Model loaded.")

    def synthesize(self, user_query, context, style=""):
        style_instruction = config.STYLES.get(style, "")
        
        # GPT-2 works better with direct dialogue completion + style hint
        style_prompt = f"Style: {style_instruction}\n" if style_instruction else ""
        prompt = (
            f"Context: {context[:500]}\n"
            f"{style_prompt}Customer: {user_query}\n"
            f"Assistant: Radhe Radhe! "
        )
        
        try:
            results = self.generator(
                prompt, 
                max_new_tokens=150 if style == "long" else 80, 
                num_return_sequences=1, 
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=50256
            )
            generated_text = results[0]['generated_text']
            # Extract just the new generated part
            response = generated_text[len(prompt):].strip()
            
            if not response or len(response) < 5:
                # Fallback to a cleaner snippet if synthesis is poor
                snippet = context.split("Snippet:")[1].split("\n")[0].strip() if "Snippet:" in context else ""
                return f"Radhe Radhe! Based on what I found: {snippet}"
                
            return f"Radhe Radhe! {response}"
        except Exception as e:
            return f"Radhe Radhe! (Local error: {e})"

class FastChatbot:
    def __init__(self, dataset_path=config.DATASET_PATH):
        if not os.path.exists(dataset_path):
             # Fallback to current directory for portability
             dataset_path = os.path.join(os.path.dirname(__file__), "dataset_refined.json")
             
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
        self.last_query = clean_query # Store for potential correction (MOVE TO START)
        normalized_query = self.clean_text(clean_query)
        query_words = set([w for w in normalized_query.split() if len(w) > 2]) # Filter very short words
        
        # 1. DL Semantic Match (The advanced DL approach)
        query_embedding = self.semantic_model.encode(clean_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.dataset_embeddings)[0]
        
        # Get the highest scoring match
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
        
        # Threshold: 0.65 for high-confidence semantic match
        if best_score > 0.65:
            # For very close matches (0.85+), return the stored response directly
            # For medium matches, use synthesizer to rephrase
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
            # 3. LLM Synthesis
            response = self.synthesizer.synthesize(clean_query, context, style)
            self.last_response = response # Store for /good reinforcement
            if response and len(response.strip()) > 5:
                return response
            else:
                print("Synthesis produced empty/short response.")
        else:
            print("Web search returned no results.")
            
        return "Radhe Radhe! I'm here to help, but I couldn't find a specific answer for that online currently. Please try asking specifically about our Sattvic food, Chitra Vrinda art, or Vrinda Tours guides!"

def main():
    bot = FastChatbot()
    
    print("\n--- Vrindopnishad Personal Assistant (Online Mode) ---")
    print("Type 'quit' or 'exit' to stop.")
    print("Available Styles: short:, long:, overview:\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # 1. Correction (/correct)
            if user_input.lower().startswith("/correct "):
                correction = user_input[9:].strip()
                if bot.last_query:
                    if bot.save_learning(bot.last_query, correction):
                        print(f"Assistant: Radhe Radhe! Learned. Next time: '{correction}'\n")
                    else:
                        print("Assistant: Processing error.\n")
                else:
                    print("Assistant: No previous query to correct.\n")
                continue

            # 2. Positive Reinforcement (/good)
            if user_input.lower() == "/good":
                if bot.last_query and bot.last_response:
                    if bot.save_learning(bot.last_query, bot.last_response):
                        print(f"Assistant: Radhe Radhe! I've baked that answer into my memory perfectly!\n")
                    else:
                        print("Assistant: Processing error.\n")
                else:
                    print("Assistant: Nothing to reinforce yet!\n")
                continue

            # 3. Trigger Local Training (/train)
            if user_input.lower() == "/train":
                print("Assistant: Synchronizing my internal Deep Learning weights with your feedback. Please wait...")
                import subprocess
                subprocess.run(["/Users/mr.bajrangi/Code/Company/.venv/bin/python", "local_tune.py"])
                # Reload model to use new weights
                bot.synthesizer = LocalSynthesizer()
                print("Assistant: Training complete! I am now smarter than before.\n")
                continue

            # 4. Check for 'wrong' prompt
            if user_input.lower() == "wrong":
                print("Assistant: I'm sorry! Please tell me the right info using: /correct [correct answer here]\n")
                continue

            # 5. Regular response
            response = bot.get_response(user_input)
            print(f"Assistant: {response}")
            print("(Type '/good' if this was great, or '/correct [answer]' if wrong!)\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
