# This script contains the full implementation of a decoder-style GPT model,

import torch
import torch.nn as nn

# Assuming the MultiHeadAttention class from Chapter 3 is in a file
# at /chapter-3/multihead_attention.py
from chapter_3.multihead_attention import MultiHeadAttention


# --- Model Configuration ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of transformer layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


# --- Building Block Implementations ---

class LayerNorm(nn.Module):
    """
    A custom Layer Normalization module.

    This module normalizes the activations of a layer to have a mean of 0
    and a variance of 1, which helps stabilize training. It includes
    learnable scale and shift parameters.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # Use unbiased=False for compatibility with original GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    This is a smooth approximation of the ReLU function and is used in
    GPT-2 and other transformer models.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    A simple feed-forward neural network module.

    It consists of two linear layers with a GELU activation in between,
    which is a standard component of the transformer block.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block.

    This block combines multi-head causal self-attention with a feed-forward
    network. It uses residual connections (shortcuts) and layer normalization
    before each main component.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


# --- Full GPT Model ---

class GPTModel(nn.Module):
    """
    The full GPT model architecture.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# --- Text Generation Function ---

def generate_text_simple(model: nn.Module, idx: torch.Tensor,
                         max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Generates text using a trained model with greedy decoding.

    Args:
        model (nn.Module): The GPT model.
        idx (torch.Tensor): The input tensor of token indices (batch, n_tokens).
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The context size supported by the model.

    Returns:
        torch.Tensor: The generated sequence of token indices.
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        
        # Get the model's predictions (logits)
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step's logits
        logits = logits[:, -1, :]
        
        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Greedily select the token with the highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append the new token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# --- Example Usage ---
if __name__ == '__main__':
    import tiktoken

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Instantiate the model with the 124M parameter configuration
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # Set model to evaluation mode

    # --- Generate text from a starting context ---
    start_context = "Every effort moves you"
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)
    print("Start context:", start_context)
    print("Encoded start context:", encoded.tolist())

    # Generate new tokens
    generated_out = generate_text_simple(
        model=model,
        idx=encoded,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # Decode the generated tokens back to text
    decoded_text = tokenizer.decode(generated_out.squeeze(0).tolist())

    print("\nGenerated text:")
    print(decoded_text)

    # --- Calculate total number of parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}") # Approx. 163M

    # Calculate parameters for GPT-2 small (124M) with weight tying
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters with weight tying: {total_params_gpt2:,}")