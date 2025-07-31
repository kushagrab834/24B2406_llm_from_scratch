# This script contains the implementation of a Multi-Head Attention module
# with causal masking, as detailed in Chapter 3. It's a key component
# for building a decoder-style transformer model like GPT.

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention module that performs scaled dot-product attention
    with causal masking.

    This implementation is designed for clarity and educational purposes,
    while remaining efficient for use in a transformer model.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_in (int): The dimensionality of the input embeddings.
            d_out (int): The dimensionality of the output. This will also be the
                         total dimension of the Q, K, V projections.
            context_length (int): The maximum length of the input sequence.
            dropout (float): The dropout probability.
            num_heads (int): The number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to the Q, K, V projections.
        """
        super().__init__()
        # d_out must be divisible by num_heads to ensure the projection
        # can be split evenly among the heads.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension of each attention head

        # Linear projections for Query, Key, and Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final output projection layer
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attention to future tokens
        # 'register_buffer' makes this a persistent part of the module,
        # but not a parameter to be trained.
        # The mask is a lower triangular matrix.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the multi-head attention.

        Args:
            x (torch.Tensor): The input tensor of shape
                              (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: The output tensor of shape
                          (batch_size, num_tokens, d_out).
        """
        batch_size, num_tokens, _ = x.shape

        # 1. Project inputs to Q, K, V
        # (batch_size, num_tokens, d_in) -> (batch_size, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. Reshape Q, K, V for multi-head attention
        # (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # 3. Transpose for matrix multiplication
        # (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. Compute scaled dot-product attention scores
        # (batch_size, num_heads, num_tokens, head_dim) @ (batch_size, num_heads, head_dim, num_tokens)
        # -> (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 5. Apply causal mask
        # We use the registered buffer, slicing it to the current sequence length.
        # This prevents attending to future positions.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6. Normalize scores to get attention weights
        # The scaling factor stabilizes the gradients.
        scaling_factor = keys.shape[-1]**0.5
        attn_weights = torch.softmax(attn_scores / scaling_factor, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Compute context vectors (weighted sum of values)
        # (batch_size, num_heads, num_tokens, num_tokens) @ (batch_size, num_heads, num_tokens, head_dim)
        # -> (batch_size, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # 8. Combine heads and project output
        # Transpose and reshape back to original batch-first format
        # (batch_size, num_heads, num_tokens, head_dim) -> (batch_size, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        # (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    d_in = 512         # Input embedding dimension
    d_out = 512        # Output dimension (must be divisible by num_heads)
    context_length = 1024 # Max sequence length
    dropout = 0.1      # Dropout rate
    num_heads = 8      # Number of attention heads
    batch_size = 4     # Number of sequences in a batch
    seq_length = 128   # Length of the example sequence

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_length, d_in)

    # Instantiate the MultiHeadAttention module
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads
    )

    # Get the output
    output = mha(dummy_input)

    # Print shapes to verify
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("\nSuccessfully executed MultiHeadAttention module.")