"""
Split LLaMA architecture for scalable curvature measurement.

This module implements a hybrid architecture that combines early layers from a
smaller LLaMA model (e.g., 8B) with late layers from a larger LLaMA model
(e.g., 70B), connected by a trainable adapter layer.

Usage:
    python demo_nanogpt.py --abc llama_split \
        --path_8b /path/to/checkpoint_8b \
        --path_70b /path/to/checkpoint_70b \
        --num_layers_8b 4 --num_layers_70b 4

The path_8b and path_70b arguments should point to directories containing
config.json (HuggingFace LlamaConfig format), or HuggingFace model IDs.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    # Standard fields (required by create_train_state interface)
    context_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embd_dim: int = 768
    bias: bool = False
    init_var: float = 1.0
    use_flash: bool = False

    # Split LLaMA specific fields
    # Paths to directories with config.json or HuggingFace model IDs
    path_8b: str = ""
    path_70b: str = ""
    num_layers_8b: int = 4
    num_layers_70b: int = 4
    use_mlp_adapter: bool = False


class GPT(nn.Module):
    """
    Split LLaMA model that combines early layers from a small model (e.g. 8B)
    with late layers from a large model (e.g. 70B), connected via a learned
    adapter that bridges the hidden dimensions.

    The forward interface matches the framework convention: forward(idx) -> logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRMSNorm,
            LlamaRotaryEmbedding,
        )

        # Load HF configs for the two model sizes
        config_8b = LlamaConfig.from_pretrained(config.path_8b)
        config_70b = LlamaConfig.from_pretrained(config.path_70b)

        attn_impl = "flash_attention_2" if config.use_flash else "eager"
        config_8b._attn_implementation = attn_impl
        config_70b._attn_implementation = attn_impl

        self._config_8b = config_8b
        self._config_70b = config_70b

        # Token embeddings (8B hidden size as the input dimension)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config_8b.hidden_size,
            padding_idx=getattr(config_8b, "pad_token_id", None),
        )

        # First set of layers (from 8B config)
        self.layers_first = nn.ModuleList(
            [
                LlamaDecoderLayer(config_8b, layer_idx=i)
                for i in range(config.num_layers_8b)
            ]
        )

        # Adapter to bridge hidden dimensions (8B -> 70B)
        if config.use_mlp_adapter:
            self.adapter_linear_1 = nn.Linear(
                config_8b.hidden_size, config_70b.hidden_size, bias=False
            )
            self.adapter_linear_2 = nn.Linear(
                config_70b.hidden_size, config_70b.hidden_size, bias=False
            )
        else:
            self.adapter = nn.Linear(
                config_8b.hidden_size, config_70b.hidden_size, bias=False
            )

        # Last set of layers (from 70B config, using the final N layers)
        start_idx_70b = config_70b.num_hidden_layers - config.num_layers_70b
        self.layers_last = nn.ModuleList(
            [
                LlamaDecoderLayer(config_70b, layer_idx=start_idx_70b + i)
                for i in range(config.num_layers_70b)
            ]
        )

        # Final normalization and language model head
        self.norm = LlamaRMSNorm(
            config_70b.hidden_size, eps=config_70b.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config_70b.hidden_size, config.vocab_size, bias=False
        )

        # Rotary embeddings for each layer group
        self.rotary_emb_8b = LlamaRotaryEmbedding(config_8b)
        self.rotary_emb_70b = LlamaRotaryEmbedding(config_70b)

    def get_num_params(self):
        """Return (total_params, embedding_params) for logging."""
        n_params = sum(p.numel() for p in self.parameters())
        embd_params = (
            self.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        )
        return n_params, embd_params

    def forward(self, idx):
        """
        Args:
            idx: Input token IDs, shape (batch_size, seq_len)

        Returns:
            logits: shape (batch_size, seq_len, vocab_size)
        """
        device = idx.device
        batch_size, seq_len = idx.shape

        hidden_states = self.embed_tokens(idx)

        position_ids = torch.arange(
            seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # 8B rotary embeddings
        position_embeddings_8b = self.rotary_emb_8b(hidden_states, position_ids)

        # Process 8B layers
        for layer in self.layers_first:
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings_8b,
            )
            hidden_states = layer_output[0]

        # Adapter: bridge 8B hidden dim -> 70B hidden dim
        if self.config.use_mlp_adapter:
            hidden_states = torch.relu(self.adapter_linear_1(hidden_states))
            hidden_states = self.adapter_linear_2(hidden_states)
        else:
            hidden_states = self.adapter(hidden_states)

        # 70B rotary embeddings
        position_embeddings_70b = self.rotary_emb_70b(hidden_states, position_ids)

        # Process 70B layers
        for layer in self.layers_last:
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings_70b,
            )
            hidden_states = layer_output[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
