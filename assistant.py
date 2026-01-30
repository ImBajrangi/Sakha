import json
import os
import re
from duckduckgo_search import DDGS
import google.generativeai as genai
import config

class WebSearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=config.MAX_SEARCH_RESULTS):
        try:
            results = self.ddgs.text(query, max_results=max_results)
            context = "\n".join([f"Source: {r['title']}\nSnippet: {r['body']}" for r in results])
            return context
        except Exception as e:
            print(f"Search error: {e}")
            return ""

class ResponseSynthesizer:
    def __init__(self, api_key=config.GOOGLE_API_KEY):
        if api_key and api_key != "YOUR_GEMINI_API_KEY":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def synthesize(self, user_query, context, style=""):
        if not self.model:
            return "I have found some information online, but I need a Gemini API Key to synthesize a better response. Here is the raw context:\n\n" + context[:500] + "..."

        style_instruction = config.STYLES.get(style, "")
        prompt = f"""
        You are a helpful assistant for Vrindopnishad (a food and pilgrimage guide for Vrindavan).
        Based on the following context, answer the user's query.
        
        {style_instruction}
        
        Context:
        {context}
        
        User Query: {user_query}
        
        Assistant Response:
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during synthesis: {e}"

class FastChatbot:
    def __init__(self, dataset_path=config.DATASET_PATH):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        self.dataset = data.get('dataset', [])
        self.searcher = WebSearcher()
        self.synthesizer = ResponseSynthesizer()
        
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
        normalized_query = self.clean_text(clean_query)
        
        # 1. Local Match
        best_match = None
        max_overlap = 0
        
        for item in self.dataset:
            instruction = self.clean_text(item['instruction'])
            if normalized_query == instruction:
                best_match = item['response']
                max_overlap = 999 # Force exact match priority
                break
            
            query_words = set(normalized_query.split())
            instr_words = set(instruction.split())
            overlap = len(query_words.intersection(instr_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = item['response']
        
        # 1b. Apply Style (if requested) or return match
        if max_overlap > 0:
            if style:
                return self.synthesizer.synthesize(clean_query, best_match, style)
            return best_match
            
        # 2. Web Search Fallback
        print(f"(Falling back to web search for: '{clean_query}')...")
        context = self.searcher.search(clean_query)
        
        if context:
            # 3. LLM Synthesis
            return self.synthesizer.synthesize(clean_query, context, style)
            
        return "Radhe Radhe! I'm here to help, but I couldn't find a specific answer for that even online. Could you please ask about Vrindopnishad, our food, art, or pilgrimage guides?"

def main():
    bot = FastChatbot()
    
    print("\n--- Vrindopnishad Personal Assistant (Online Mode) ---")
    print("Type 'quit' or 'exit' to stop.")
    print("Available Styles: short:, long:, overview:\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            response = bot.get_response(user_input)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
