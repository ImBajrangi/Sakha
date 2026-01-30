import torch
import json
import os
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

def train():
    output_dir = "/Users/mr.bajrangi/Code/Company/Projects/Sakha/model_output"
    model_name = "distilgpt2" 
    dataset_path = "/Users/mr.bajrangi/Code/Company/Projects/Sakha/dataset_refined.json"

    print("Loading raw dataset...")
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)
    samples = raw_data.get('dataset', [])

    # Duplicate dataset significantly to force memorization
    expanded_samples = []
    for _ in range(100): 
        expanded_samples.extend(samples)

    print(f"Total samples after expansion: {len(expanded_samples)}")

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
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            
            full_ids = prompt_ids + response_ids
            # Truncate if too long (unlikely for this data)
            full_ids = full_ids[:128]
            
            # Mask prompt ids in labels
            labels = [-100] * len(prompt_ids) + response_ids
            labels = labels[:128]
            
            # Padding
            padding_length = 128 - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids_list]
        }

    ds = Dataset.from_list(expanded_samples)
    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50, 
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=50,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        report_to="none",
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )

    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()
