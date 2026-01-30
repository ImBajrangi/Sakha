import torch
import logging
from transformers import pipeline
import config

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
