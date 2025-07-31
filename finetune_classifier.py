# This script contains the complete pipeline for finetuning a pretrained
# GPT model for a text classification task (spam vs. not spam).

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Import from previous chapters ---
# Assuming these files are in the specified directory structure
from chapter_4.gpt_model import GPTModel, GPT_CONFIG_124M
from chapter_5.gpt_download import download_and_load_gpt2
from chapter_5.train_and_generate import load_weights_into_gpt, text_to_token_ids

# --- Dataset Preparation ---

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

# --- DataLoader Creation ---

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [text[:self.max_length] for text in self.encoded_texts]

        self.encoded_texts = [
            text + [pad_token_id] * (self.max_length - len(text))
            for text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(text) for text in self.encoded_texts)

# --- Training and Evaluation Functions for Classification ---

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Use only the logits of the last token for classification
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
        
    return correct_predictions / num_examples

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# --- Plotting ---
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.show()

# --- Inference ---
def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    encoded = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    
    encoded = encoded[:min(max_length, supported_context_length)]
    padded = encoded + [pad_token_id] * (max_length - len(encoded))
    input_tensor = torch.tensor(padded, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. Dataset Preparation ---
    df = pd.read_csv("sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # --- 2. DataLoader Creation ---
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # --- 3. Model Adaptation ---
    BASE_CONFIG = GPT_CONFIG_124M.copy()
    BASE_CONFIG.update({
        "context_length": train_dataset.max_length,
        "qkv_bias": True # Required for pretrained weights
    })
    
    model_size = "124M"
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze and replace the output head
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    
    # Unfreeze the last transformer block and final layer norm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 4. Finetuning ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 5
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

    # --- 5. Evaluation and Plotting ---
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Final accuracies: Train({train_accuracy*100:.2f}%) | Val({val_accuracy*100:.2f}%) | Test({test_accuracy*100:.2f}%)")

    # --- 6. Inference ---
    print("\n--- Inference Examples ---")
    text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
    print(f"'{text_1}' is {classify_review(text_1, model, tokenizer, device, train_dataset.max_length)}")

    text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    print(f"'{text_2}' is {classify_review(text_2, model, tokenizer, device, train_dataset.max_length)}")
