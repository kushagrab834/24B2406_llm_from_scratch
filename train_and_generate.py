# This script contains the functions for training the GPT model,
# evaluating its performance, and generating text with advanced
# decoding strategies like temperature scaling and top-k sampling.
# It also includes the logic to load pretrained GPT-2 weights from OpenAI.

import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Assuming the GPTModel class from Chapter 4 is in a file
# at /chapter-4/gpt_model.py
from chapter_4.gpt_model import GPTModel, GPT_CONFIG_124M

# --- Text Generation with Decoding Strategies ---

def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int,
             context_size: int, temperature: float = 0.0, top_k: int = None,
             eos_id: int = None) -> torch.Tensor:
    """
    Generates text using a trained model with temperature scaling and top-k sampling.

    Args:
        model (nn.Module): The GPT model.
        idx (torch.Tensor): Input tensor of token indices (batch, n_tokens).
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): The context size supported by the model.
        temperature (float): Controls randomness. Higher values increase diversity.
                             A value of 0.0 makes it deterministic (greedy).
        top_k (int, optional): If set, samples from the top 'k' most likely tokens.
        eos_id (int, optional): If specified, stops generation if this token is produced.

    Returns:
        torch.Tensor: The generated sequence of token indices.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# --- Loss Calculation Utilities ---

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor,
                    model: nn.Module, device: torch.device) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model: nn.Module, device: torch.device,
                     num_batches: int = None) -> float:
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# --- Training and Evaluation Functions ---

def evaluate_model(model: nn.Module, train_loader, val_loader,
                   device: torch.device, eval_iter: int):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model: nn.Module, tokenizer, device: torch.device,
                              start_context: str):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model: nn.Module, train_loader, val_loader, optimizer,
                       device: torch.device, num_epochs: int, eval_freq: int,
                       eval_iter: int, start_context: str, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

# --- Utility and Plotting Functions ---

def text_to_token_ids(text: str, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()

# --- Pretrained Weight Loading ---

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt: GPTModel, params: dict):
    import numpy as np
    
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# --- Main Execution ---
if __name__ == '__main__':
    # NOTE: This main block is for demonstration. Training from scratch is
    # computationally intensive and is best run on a machine with a GPU.
    # The second part demonstrates loading pretrained weights, which is
    # much more practical on a standard computer.

    # --- Part 1: Train a model from scratch (optional, requires GPU) ---
    print("Part 1: Training a model from scratch...")
    # Code for training would go here, including data loading, etc.
    # Due to its length and resource requirements, it's omitted from this
    # example block but is fully detailed in the notebook.
    # The key components are the functions defined above.
    print("Skipping training from scratch in this example.")


    # --- Part 2: Load pretrained GPT-2 weights and generate text ---
    print("\nPart 2: Loading pretrained GPT-2 weights...")
    
    # Import the download utility (assumes gpt_download.py is in the same directory)
    from gpt_download import download_and_load_gpt2

    # Define model configurations
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    model_name = "gpt2-small (124M)"
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    # Update configuration to match pretrained model
    pretrained_config = GPT_CONFIG_124M.copy()
    pretrained_config.update(model_configs[model_name])
    pretrained_config.update({"context_length": 1024, "qkv_bias": True})

    # Instantiate and load weights
    gpt = GPTModel(pretrained_config)
    load_weights_into_gpt(gpt, params)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)
    gpt.eval()

    # Generate text with the pretrained model
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    print(f"\nGenerating text with pretrained {model_name}...")
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=encoded,
        max_new_tokens=25,
        context_size=pretrained_config["context_length"],
        top_k=50,
        temperature=1.5
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f'"{start_context}{decoded_text[len(start_context):]}"')
