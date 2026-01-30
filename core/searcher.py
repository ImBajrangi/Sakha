import re
import config
import time
from duckduckgo_search import DDGS

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
                    time.sleep(2)
                    continue
                break
        return ""
