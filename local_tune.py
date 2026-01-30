import torch
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import config

def run_local_tune(epochs=10):
    print(f"--- Starting Local RLHF Fine-Tuning (Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}) ---")
    
    # Load latest dataset
    if not os.path.exists(config.DATASET_PATH):
        print("No dataset found to train on.")
        return

    with open(config.DATASET_PATH, 'r') as f:
        data = json.load(f)
    samples = data.get('dataset', [])
    
    if len(samples) < 5:
        print("Dataset too small for effective fine-tuning (< 5 samples).")
        return

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenization
    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        for q, a in zip(examples['instruction'], examples['response']):
            prompt, response = f"Customer: {q}\nAssistant: ", f"Radhe Radhe! {a}{tokenizer.eos_token}"
            p_ids, r_ids = tokenizer.encode(prompt, add_special_tokens=False), tokenizer.encode(response, add_special_tokens=False)
            full_ids = (p_ids + r_ids)[:128]
            labels = ([-100] * len(p_ids) + r_ids)[:128]
            pad_len = 128 - len(full_ids)
            full_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids_list]}

    # Expand slightly for better convergence on small sets
    expanded = []
    for _ in range(5): 
        expanded.extend(samples)

    ds = Dataset.from_list(expanded)
    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

    # Training Args (Optimized for Mac MPS)
    training_args = TrainingArguments(
        output_dir=config.MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_strategy="no",
        learning_rate=2e-4,
        weight_decay=0.01,
        use_mps_device=torch.backends.mps.is_available(),
        report_to="none",
        logging_steps=10
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
    trainer.train()
    
    # Save back to same directory
    trainer.save_model(config.MODEL_PATH)
    tokenizer.save_pretrained(config.MODEL_PATH)
    print("\nSUCCESS: Model weights synchronized with latest RLHF feedback!")

if __name__ == "__main__":
    run_local_tune()
