# This script contains the complete pipeline for instruction-finetuning a
# pretrained GPT model. It covers dataset preparation, creating custom
# data loaders with dynamic padding, and the finetuning process itself.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import os
import urllib
from functools import partial

# --- Import from previous chapters ---
from chapter_4.gpt_model import GPTModel
from chapter_5.gpt_download import download_and_load_gpt2
from chapter_5.train_and_generate import (
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    generate,
    calc_loss_loader,
    train_model_simple
)

# --- Dataset Preparation ---

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def format_input(entry: dict) -> str:
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# --- DataLoader Creation ---

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # Add an end-of-sequence token
            full_text += "<|endoftext|>"
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, device="cpu"):
    batch_max_length = max(len(item) for item in batch)
    
    inputs_lst, targets_lst = [], []
    for item in batch:
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        # Mask padding tokens
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices] = ignore_index
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. Dataset Preparation ---
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # --- 2. Load Pretrained Model ---
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    model_configs = {
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" (")[1].rstrip(")")
    
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # --- 3. Create DataLoaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    customized_collate_fn = partial(custom_collate_fn, device=device)
    
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=customized_collate_fn, shuffle=True, drop_last=True)
    
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=customized_collate_fn, shuffle=False)

    # --- 4. Finetuning ---
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 2
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    # --- 5. Save the finetuned model ---
    model_name_safe = CHOOSE_MODEL.replace(" ", "-").replace("(", "").replace(")", "")
    file_name = f"{model_name_safe}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")