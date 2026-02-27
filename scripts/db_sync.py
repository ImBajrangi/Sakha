import os
import json
from supabase import create_client, Client
import dotenv

# Load environment variables (pulling from Sakha's own context or hardcoded from discovery)
# USER PROVIDED SUPABASE CONFIG
SUPABASE_URL = "https://tilimltxgeucefxzerqi.supabase.co"
SUPABASE_KEY = "sb_publishable_0YiM-Q8itRORUDdToracaQ_vzcrjUlC"

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")

def sync_supabase():
    print(f"Radhe Radhe! Connecting to Supabase: {SUPABASE_URL}...")
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Fetch content from the 'content' table
        print("Fetching spiritual articles from 'content' table...")
        response = supabase.table("content").select("*").execute()
        
        if not response.data:
            print("No data found in Supabase.")
            return

        print(f"Retrieved {len(response.data)} articles. Mapping to Sakha dataset...")
        
        new_entries = []
        for item in response.data:
            title = item.get('title', 'Unknown Title')
            hindi = item.get('hindi_text', '')
            english = item.get('english_translation', '')
            category = item.get('category', 'Spirituality')
            
            # Create a combined knowledge pair
            instruction = f"Show me {title}"
            
            # Sakha likes factual, structured responses
            res_parts = [f"Radhe Radhe! This is from our {category} collection."]
            if hindi:
                res_parts.append(f"**Hindi**: {hindi}")
            if english:
                res_parts.append(f"**English**: {english}")
            
            response_text = "\n\n".join(res_parts)
            
            new_entries.append({
                "instruction": instruction,
                "response": response_text
            })
            
            # Also add a "tell me about" variation
            new_entries.append({
                "instruction": f"Tell me about {title}",
                "response": response_text
            })

        update_dataset(new_entries)

    except Exception as e:
        print(f"Error during Supabase sync: {e}")

def update_dataset(new_data):
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # Deduplicate by instruction to avoid bloating
    existing_instructions = {item['instruction'].lower() for item in data['dataset']}
    
    count = 0
    for item in new_data:
        if item['instruction'].lower() not in existing_instructions:
            data['dataset'].append(item)
            existing_instructions.add(item['instruction'].lower())
            count += 1

    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Radhe Radhe! Successfully ingested {count} new entries from Supabase into Sakha's brain.")

if __name__ == "__main__":
    sync_supabase()
