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
        self.last_subject = "" # Track the last mentioned project/brand subject
        self.last_response = ""
        
        # Initialize DL Semantic Model
        print("Loading DL Semantic Model (Sentence-BERT)...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset_embeddings = None
        self._update_embeddings()

        # Navigation Map (Keyword: (URL, Name, Priority))
        # Higher priority wins for compound queries like "vrindopnishad kitchen"
        self.NAV_MAP = {
            'kitchen': ('https://vrindopnishad.in/Projects/Cloud-Kitchen/kitchen.html', 'Satvik Kitchen', 10),
            'foody': ('https://vrindopnishad.in/Projects/Cloud-Kitchen/kitchen.html', 'Satvik Kitchen', 10),
            'food': ('https://vrindopnishad.in/Projects/Cloud-Kitchen/kitchen.html', 'Satvik Kitchen', 10),
            'vrinda tours': ('https://vrindopnishad.in/Projects/Vrinda-Tours/vrinda-tours.html', 'Vrinda Tours', 15),
            'tour': ('https://vrindopnishad.in/Projects/Vrinda-Tours/vrinda-tours.html', 'Vrinda Tours', 10),
            'tours': ('https://vrindopnishad.in/Projects/Vrinda-Tours/vrinda-tours.html', 'Vrinda Tours', 10),
            'yatra': ('https://vrindopnishad.in/Projects/Vrinda-Tours/vrinda-tours.html', 'Brij Yatra', 10),
            'skillTadka': ('https://edu.vrindopnishad.in/', 'skillTadka Library', 10),
            'library': ('https://edu.vrindopnishad.in/', 'skillTadka Library', 10),
            'chitra vrinda': ('https://vrindopnishad.in/Vrindopnishad%20Web/Pictures/main/Gallery.html', 'Chitra Vrinda', 15),
            'chitra': ('https://vrindopnishad.in/Vrindopnishad%20Web/Pictures/main/Gallery.html', 'Chitra Vrinda', 10),
            'art': ('https://vrindopnishad.in/Vrindopnishad%20Web/Pictures/main/Gallery.html', 'Chitra Vrinda', 10),
            'gallery': ('https://vrindopnishad.in/Vrindopnishad%20Web/Pictures/main/Gallery.html', 'Chitra Vrinda', 10),
            'vrinda vaani': ('https://www.youtube.com/@vrindopnishad', 'Vrinda Vaani', 15),
            'vaani': ('https://www.youtube.com/@vrindopnishad', 'Vrinda Vaani', 10),
            'sant-vaani': ('https://www.youtube.com/@vrindopnishad', 'Sant-Vaani', 10),
            'instagram': ('https://www.instagram.com/vrindopnishad/', 'Instagram', 10),
            'insta': ('https://www.instagram.com/vrindopnishad/', 'Instagram', 10),
            'youtube': ('https://www.youtube.com/@vrindopnishad', 'YouTube', 10),
            'channel': ('https://www.youtube.com/@vrindopnishad', 'YouTube Channel', 10),
            'facebook': ('https://www.facebook.com/vrindopnishad/', 'Facebook', 10),
            'fb': ('https://www.facebook.com/vrindopnishad/', 'Facebook', 10),
            'whatsapp': ('https://whatsapp.com/channel/0029Vb6UR3Z9mrGcDXbHzA1Q', 'WhatsApp Channel', 10),
            'darshan': ('https://chat.whatsapp.com/LUMjP73wwyY9C1DNYeyoGu', 'Daily Darshan Group', 10),
            'group': ('https://chat.whatsapp.com/LUMjP73wwyY9C1DNYeyoGu', 'Daily Darshan Group', 10),
            'home': ('https://vrindopnishad.in/', 'Vrindopnishad Home', 1),
            'vrindopnishad': ('https://vrindopnishad.in/', 'Vrindopnishad Home', 1)
        }

    def intercept_navigation(self, query):
        """Intercept queries with navigation intent."""
        nav_triggers = ['open', 'go to', 'visit', 'take me to', 'link of', 'navigate to', 'show me']
        query_lower = query.lower().strip()
        
        # Find all matches and pick by priority then length
        matches = []
        for key, (url, name, priority) in self.NAV_MAP.items():
            if key in query_lower:
                matches.append((key, url, name, priority))
        
        if matches:
            # Sort by priority DESC, then length of key DESC
            matches.sort(key=lambda x: (x[3], len(x[0])), reverse=True)
            best_key, url, name, priority = matches[0]
            
            # If it's a very short query or specifically has a trigger
            has_trigger = any(t in query_lower for t in nav_triggers)
            if has_trigger or len(query_lower.split()) <= 3:
                 return f"Radhe Radhe! Opening {name} for you... [NAV: {url}]"
        
        return None

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

    def contextualize_query(self, query, history=None):
        """Enrich short, contextual queries like 'link?' with subject from history."""
        q = query.lower().strip()
        # Common follow-up patterns
        link_patterns = [r'^link\??$', r'^website\??$', r'^url\??$', r'^site\??$']
        info_patterns = [r'^where\??$', r'^founder\??$', r'^who\??$', r'^details\??$']
        
        is_followup = any(re.match(p, q) for p in link_patterns + info_patterns)
        
        if is_followup:
            # Try to resolve subject from provided history first
            brand_keywords = ['foody', 'skillTadka', 'chitra', 'vaani', 'kitchen', 'tour', 'yatra', 'social', 'media', 'instagram', 'youtube', 'whatsapp', 'facebook', 'vrindopnishad', 'links', 'handles', 'sakha', 'vrinda', 'brij', 'hariom', 'yash', 'dhani', 'krishna', 'radhe']
            subject = self.last_subject
            
            if history:
                # Scan history from latest to oldest for any brand keyword
                for turn in reversed(history):
                    content = turn.get('content', '').lower()
                    for k in brand_keywords:
                        if k in content:
                            subject = k
                            break
                    if subject: break

            if subject:
                if any(re.match(p, q) for p in link_patterns):
                    return f"{subject} link"
                return f"{subject} {q}"
        
        return query

    def parse_style(self, user_query):
        # Check for style prefix like "short: " or "overview: "
        for style in config.STYLES:
            if user_query.lower().startswith(f"{style}:"):
                return style, user_query[len(style)+1:].strip()
        return None, user_query

    def get_response(self, user_query, history=None):
        # 0. Navigation Interception
        nav_response = self.intercept_navigation(user_query)
        if nav_response:
            return {"response": nav_response, "navigate": re.search(r'\[NAV: (https?://.*?)\]', nav_response).group(1)}

        style, raw_query = self.parse_style(user_query)
        clean_query = self.contextualize_query(raw_query, history=history)
        self.last_query = raw_query 

        # 1. Classification: Is this about our brand?
        brand_keywords = ['foody', 'skillTadka', 'chitra', 'vaani', 'kitchen', 'tour', 'yatra', 'social', 'media', 'instagram', 'youtube', 'whatsapp', 'facebook', 'vrindopnishad', 'links', 'handles', 'sakha', 'vrinda', 'brij', 'hariom', 'yash', 'dhani', 'krishna', 'radhe']
        is_brand_query = any(k in clean_query.lower() for k in brand_keywords)
        # 2. Semantic Match Check
        query_embedding = self.semantic_model.encode(clean_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.dataset_embeddings)[0]
        
        # 2.1 Keyword Boosting: If query contains a brand keyword, boost matches that also contain it
        boosted_scores = cos_scores.clone()
        for i, item in enumerate(self.dataset):
            instr = item['instruction'].lower()
            for k in brand_keywords:
                if k in clean_query.lower() and k in instr:
                    boosted_scores[i] += 0.15 # Strong boost for shared specific brand keywords
        
        best_idx = torch.argmax(boosted_scores).item()
        best_score = boosted_scores[best_idx].item()

        print(f"(Query: '{clean_query}', Score: {best_score:.2f}, Original: {cos_scores[best_idx].item():.2f}, Brand: {is_brand_query})")

        response_text = ""
        navigate_url = None

        # 3. Decision Logic
        if best_score > 0.85 or (is_brand_query and best_score > 0.60):
            response_text = self.dataset[best_idx]['response']
            print(f" -> Using Local Match (Logic A)")
            
            # Update last subject
            matched_instr = self.dataset[best_idx]['instruction'].lower()
            for k in brand_keywords:
                if k in matched_instr:
                    self.last_subject = k
                    break

            # BRAND PROTECTION: If it's a high-confidence brand match, don't let GPT-2 hallucinate
            # Only synthesize if style is requested AND either it's not brand or score is low
            if style:
                should_synthesize = True
                if is_brand_query and best_score > 0.80 and style != "long":
                    should_synthesize = False
                
                if should_synthesize:
                    response_text = self.synthesizer.synthesize(clean_query, response_text, style, history=history)
                    
        elif is_brand_query and 0.4 < best_score <= 0.60:
            # Proactive Clarification: Broad brand query but not a strong match
            print(f" -> Proactive Clarification triggered.")
            matched_instr = self.dataset[best_idx]['instruction'].lower()
            topic = "our services"
            for k in brand_keywords:
                if k in clean_query.lower():
                    topic = k
                    break
            response_text = f"Radhe Radhe! I see you're asking about '{topic}'. Could you clarify what exactly you'd like to know? You can ask for our menu, gallery, or pilgrimage details."
            
        else:
            # Case B: General Knowledge fallback
            # If the score is extremely low, it's completely out of domain. Skip web search and offer contextual guidance.
            if best_score < 0.25:
                print(f" -> Very low score ({best_score:.2f}). Triggering Contextual Fallback directly.")
                context = None
            else:
                print(f" -> Falling back to web search (Logic B)...")
                context = self.searcher.search(clean_query)
            
            if context:
                print(f"Web search found context ({len(context)} chars). Synthesizing...")
                response_text = self.synthesizer.synthesize(clean_query, context, style, history=history)
            else:
                # Contextual Fallback
                if self.last_subject in ['food', 'foody', 'kitchen']:
                    response_text = "Radhe Radhe! I'm not sure about that. Were you trying to order Sattvic food? You can simply say 'open kitchen'."
                elif self.last_subject in ['tour', 'tours', 'yatra']:
                    response_text = "Radhe Radhe! I'm not sure about that. Are you looking for Brij Yatra guides? You can say 'open vrinda tours'."
                elif self.last_subject in ['art', 'chitra', 'gallery']:
                    response_text = "Radhe Radhe! I'm not sure about that. Want to explore our divine artwork? Just say 'open chitra vrinda'."
                elif self.last_subject == 'skillTadka':
                    response_text = "Radhe Radhe! I'm not sure about that. Need some spiritual reading? Say 'visit skillTadka library'."
                else:
                    response_text = "Radhe Radhe! I'm still learning and don't have an answer for that yet. How can I help you with our Sattvic kitchen, spiritual art, or Vrinda Tours?"

        # Extract navigation markers like [NAV: https://...]
        nav_match = re.search(r'\[NAV: (https?://.*?)\]', response_text)
        if nav_match:
            navigate_url = nav_match.group(1).strip()
            response_text = response_text.replace(nav_match.group(0), "").strip()

        return {"response": response_text, "navigate": navigate_url}
