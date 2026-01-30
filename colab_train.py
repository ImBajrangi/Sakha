# STANDALONE COLAB TRAINING SCRIPT (PERFECT RESULTS VERSION)
# To run: Open a new notebook in Google Colab (colab.research.google.com), 
# Menu -> Runtime -> Change runtime type -> Select T4 GPU.
# Copy-paste this entire script into a code cell and run it.

import torch
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# 1. THE "RIGHT INFORMATION" (Refined Dataset)
dataset_json = {
  "dataset": [
    {"instruction": "What is Vrindopnishad?", "response": "Radhe Radhe! Vrindopnishad is a digital sanctuary bridging Vedic wisdom with modern innovation. We offer spiritual art, Sattvic food, pilgrimage guides, and a PDF library of scriptures. Visit vrindopnishad.in to learn more."},
    {"instruction": "What services do you offer?", "response": "Jai Shri Krishna! We offer: 1. Foody Vrinda (Sattvic kitchen), 2. Chitra Vrinda (Art gallery), 3. Vrinda Tours (Brij Yatra guides), 4. PDF Library (Scriptures), and 5. Vrinda Chat (Community messenger)."},
    {"instruction": "Tell me about Foody Vrinda.", "response": "Foody Vrinda is our 100% Sattvic cloud kitchen. We serve pure vegetarian meals without onion or garlic, prepared with devotion in Vrindavan."},
    {"instruction": "Is the food from Foody Vrinda vegetarian?", "response": "Yes, all meals are strictly vegetarian, Sattvic, and prepared without onion or garlic, following the highest standards of spiritual purity."},
    {"instruction": "What is Chitra Vrinda?", "response": "Chitra Vrinda is our digital art gallery. It features diverse collections including Traditional Vedic art, Anime-inspired spiritual themes, Warrior styles, and cute Chhibi renditions of divine figures."},
    {"instruction": "What is Vrinda Tours?", "response": "Vrinda Tours is an interactive guide for the Brij 84 Kos Yatra. It provides live maps, temple timings, and personalized itineraries for pilgrims visiting Vrindavan and surrounding holy sites."},
    {"instruction": "Can you help me plan a trip to Vrindavan?", "response": "Absolutely! Visit our Vrinda Tours section for interactive maps, temple intel, and Yatra plans for locations like Banke Bihari, Prem Mandir, and the 84 Kos Parikrama."},
    {"instruction": "Do you have any books or scriptures?", "response": "Yes, we have a free PDF Library featuring Shri Harivansh literature, the Bhagavad Gita, Upanishads, and teachings from various saints. You can download them at vrindopnishad.in."},
    {"instruction": "What is Sant-Vaani?", "response": "Sant-Vaani is a collection of sacred shlokas and teachings from revered saints like Premanand Ji Maharaj, aimed at providing daily spiritual inspiration."},
    {"instruction": "Who is Premanand Ji Maharaj?", "response": "Radhe Radhe! Sant Shri Premanand Ji Maharaj is a revered saint in Vrindavan known for his profound satsangs on pure devotion (Nitya-Vihar) and the glory of Shri Radha."},
    {"instruction": "What is Vrinda Chat?", "response": "Vrinda Chat is our community messenger for devotees. It supports private and group messaging, media sharing, and easy connection via QR code scanning."},
    {"instruction": "How do I share my ID on Vrinda Chat?", "response": "Open Vrinda Chat and click 'Share ID'. This generates a unique QR code that others can scan to start a private chat with you."},
    {"instruction": "How can I contact you?", "response": "You can reach us via our feedback portal on vrindopnishad.in, our Instagram @vrindopnishad, or our official WhatsApp channel."},
    {"instruction": "Is my data safe with you?", "response": "Yes, we value your privacy. We implement secure protocols across our platform, including Vrinda Chat. Details are in our 2026 Privacy Policy on our website."},
    {"instruction": "What is the 84 Kos Yatra?", "response": "The Brij 84 Kos Yatra is a 168-mile pilgrimage visiting sites associated with Radha and Krishna. Vrinda Tours provides a digital companion map for this entire journey."},
    {"instruction": "Who is Shri Harivansh?", "response": "Shri Hit Harivansh Mahaprabhu is the founder of the Radha Vallabh Sampradaya. Our library includes many texts dedicated to his life and teachings."},
    {"instruction": "What are Chhibi Styles in your gallery?", "response": "Chhibi Styles are cute, simplified artistic representations of spiritual figures in our Chitra Vrinda gallery, perfect for a playful expression of devotion."},
    {"instruction": "Radhe Radhe!", "response": "Radhe Radhe! Welcome to our digital sanctuary. How can I assist you with our services in art, food, or scriptures today?"},
    {"instruction": "Jai Shri Krishna!", "response": "Jai Shri Krishna! I am here to help you explore Vrindopnishad. What can I do for you today?"},
    {"instruction": "Thank you so much!", "response": "Radhe Radhe! It is my joy to be of service. Please feel free to return if you have more questions. May your day be filled with peace!"}
  ]
}

def train_colab():
    output_dir = "./model_output"
    model_name = "gpt2" # Using standard GPT2 for more capacity
    samples = dataset_json["dataset"]

    # 2. INTENSE EXPANSION (100x)
    expanded_samples = []
    for _ in range(100):
        expanded_samples.extend(samples)

    print(f"Total samples for training: {len(expanded_samples)}")

    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        for q, a in zip(examples['instruction'], examples['response']):
            prompt = f"Customer: {q}\nAssistant: "
            response = f"{a}{tokenizer.eos_token}"
            
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            r_ids = tokenizer.encode(response, add_special_tokens=False)
            
            full_ids = p_ids + r_ids
            full_ids = full_ids[:128]
            
            # Mask prompt ids in labels (set to -100)
            labels = ([-100] * len(p_ids)) + r_ids
            labels = labels[:128]
            
            # Padding
            pad_len = 128 - len(full_ids)
            full_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            
        return {
            "input_ids": input_ids_list, 
            "labels": labels_list,
            "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids_list]
        }

    ds = Dataset.from_list(expanded_samples)
    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

    # 3. HIGH RATE TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=50, # 50 epochs on 100x data = 5000 effective iterations
        per_device_train_batch_size=8,
        save_strategy="no",
        learning_rate=5e-4, # High rate for fast convergence on tiny data
        weight_decay=0.01,
        fp16=True, 
        report_to="none",
        logging_steps=100
    )

    print("Starting intense training...")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Create zip for easy downloading
    import shutil
    shutil.make_archive("model_output", 'zip', output_dir)
    print("\nSUCCESS! Download 'model_output.zip' from the files sidebar.")

if __name__ == "__main__":
    train_colab()
