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

    def synthesize(self, user_query, context, style="", history=None):
        style_instruction = config.STYLES.get(style, "")
        
        # Format history turns (last 3 turns to fit context)
        history_text = ""
        if history:
            for turn in history[-3:]:
                role = "Customer" if turn.get('role') == 'user' else "Assistant"
                history_text += f"{role}: {turn.get('content')}\n"

        # GPT-2 works better with direct dialogue completion + style hint
        style_prompt = f"Style: {style_instruction}\n" if style_instruction else ""
        prompt = (
            f"Context: {context[:500]}\n"
            f"{history_text}"
            f"{style_prompt}Customer: {user_query}\n"
            f"Assistant: Radhe Radhe! "
        )
        try:
            # print(f"DEBUG - Context Length: {len(context)}")
            # If we have web context, we prefer the snippet unless the model produces something very long and unique
            results = self.generator(
                prompt, 
                max_new_tokens=150 if style == "long" else 80, 
                num_return_sequences=1, 
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.5, # Increased
                pad_token_id=50256
            )
            generated_text = results[0]['generated_text']
            # Extract just the new generated part
            response = generated_text[len(prompt):].strip()
            
            # STRENGHTENED FALLBACK: If we have context, prioritize the snippet for general queries
            if context:
                snippet_match = re.search(r'Snippet: (.*?)(?:\nSource:|$)', context, re.DOTALL)
                if snippet_match:
                    snippet = snippet_match.group(1).strip()
                    # If local model is repeating itself or is much shorter than the snippet, use snippet
                    if len(response) < 20 or response.count(response[:5]) > 2:
                        response = snippet
                    # Also use snippet if local model hallucinations detected (mentioning brand keywords incorrectly)
                    elif "84 kos" in response.lower() and "yatra" not in user_query.lower():
                        response = snippet
            
            if not response:
                return "Radhe Radhe! I'm still learning about this, but I'll try to find more information soon."
                
            return f"Radhe Radhe! {response}"
        except Exception as e:
            import re
            snippet_match = re.search(r'Snippet: (.*?)(?:\nSource:|$)', context, re.DOTALL)
            if snippet_match:
                return f"Radhe Radhe! {snippet_match.group(1).strip()}"
            return f"Radhe Radhe! (Local error: {e})"
