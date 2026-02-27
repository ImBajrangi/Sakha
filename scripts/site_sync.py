import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time

# VRINDOPNISHAD SITE CONFIG
BASE_URL = "https://vrindopnishad.in"
PROJECTS = [
    "/",
    "/Projects/Cloud-Kitchen/kitchen.html",
    "/Projects/Vrinda-Tours/vrinda-tours.html",
    "/Projects/Brij%20Yatra/vrinda%20yatra.html",
    "/Vrindopnishad%20Web/Pictures/main/Gallery.html",
    "/Projects/Web%20dev/vrinda%20web%20dev.html",
    "/Projects/3DWeb/index.html"
]

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")

def clean_text(text):
    # Remove extra whitespace and newlines
    return re.sub(r'\s+', ' ', text).strip()

def crawl_site():
    print(f"Radhe Radhe! Starting automated sync for {BASE_URL}...")
    knowledge_base = []
    
    headers = {
        'User-Agent': 'SakhaKnowledgeBot/1.0 (+https://vrindopnishad.in)'
    }

    for path in PROJECTS:
        # Use properly escaped path
        url = BASE_URL + path.replace(" ", "%20")
        print(f"Crawling: {url}...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()

            # Extract headers and paragraphs
            content = []
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
                txt = clean_text(tag.get_text())
                if len(txt) > 20:
                    content.append(txt)

            # Deduplicate and limit
            unique_content = list(dict.fromkeys(content))
            
            # Add to local knowledge
            for text in unique_content:
                knowledge_base.append({
                    "instruction": f"Tell me about {path.split('/')[-1].replace('.html', '') if '/' in path else 'Vrindopnishad'}",
                    "response": f"Radhe Radhe! From our website: {text}"
                })

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    return knowledge_base

def update_dataset(new_data):
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # Simple deduplication by instruction
    existing_instructions = {item['instruction'].lower() for item in data['dataset']}
    
    count = 0
    for item in new_data:
        if item['instruction'].lower() not in existing_instructions:
            data['dataset'].append(item)
            existing_instructions.add(item['instruction'].lower())
            count += 1

    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Radhe Radhe! Ingested {count} new knowledge points into Sakha's brain.")

if __name__ == "__main__":
    new_knowledge = crawl_site()
    if new_knowledge:
        update_dataset(new_knowledge)
