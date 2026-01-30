import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_chatbot():
    model_path = "/Users/mr.bajrangi/Code/Company/Projects/CustomerAssistantBot/model_output"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    queries = [
        "Who is Premanand Ji Maharaj?",
        "What are Chhibi Styles?",
        "Are the meals in your kitchen vegetarian?",
        "How do I share my ID on Vrinda Chat?",
        "Tell me about the 84 Kos Yatra.",
        "Radhe Radhe!"
    ]

    print("\n--- Testing Chatbot Responses ---\n")
    for query in queries:
        formatted_prompt = f"Customer: {query}\nAssistant:"
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs, 
            max_length=300, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Split by Assistant prefix and then by EOS token to get exactly the assistant's part
        assistant_part = response.split("Assistant:")[-1].split("<|endoftext|>")[0].strip()
        print(f"Query: {query}")
        print(f"Response: {assistant_part}\n")

if __name__ == "__main__":
    test_chatbot()
