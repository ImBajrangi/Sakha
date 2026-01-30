import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_chatbot(model_path):
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Move to GPU if available (MPS for Mac)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, max_length=100):
    formatted_prompt = f"Customer: {prompt}\nAssistant:"
    
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=3, # Increased to 3 to handle longer spiritual phrases better
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.85 # Slightly higher for more varied vocabulary usage
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's part
    assistant_part = response.split("Assistant:")[-1].split("Customer:")[0].strip()
    return assistant_part

def main():
    model_path = "/Users/mr.bajrangi/Code/Company/Projects/CustomerAssistantBot/model_output"
    model, tokenizer, device = load_chatbot(model_path)
    
    print("\n--- Customer Assistant Chatbot ---")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        response = generate_response(model, tokenizer, device, user_input)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()
