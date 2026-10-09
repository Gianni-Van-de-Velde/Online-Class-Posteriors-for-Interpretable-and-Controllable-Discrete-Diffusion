# AMPLIFY (Fournier et al., 2024), https://github.com/chandar-lab/AMPLIFY, adapted for discrete
# flow matching: the model is conditioned on the noise level t and on a V-gene class label, via
# AdaLN blocks taken from DiT (https://github.com/facebookresearch/DiT).
# Rotary embeddings follow https://github.com/facebookresearch/llama/blob/main/llama/model.py.
import math

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from xformers.ops import SwiGLU

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput

from .rotary import precompute_freqs_cis, apply_rotary_emb


class AMPLIFYConfig(PretrainedConfig):
    model_type = "AMPLIFY"

    # All config parameters must have a default value (HuggingFace requirement).
    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 3840,
        frequency_embedding_size: int = 256,   # width of the sinusoidal timestep features
        max_period: float = 10,                # see TimestepEmbedder: the released runs use 10
        dropout_prob: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-05,
        hidden_act: str = "SwiGLU",
        layer_norm_after_embedding: bool = False,
        layer_norm_before_last_layer: bool = True,
        layer_norm_eps: float = 1e-05,
        vocab_size: int = 27,
        ffn_bias: bool = False,
        att_bias: bool = False,
        pad_token_id: int = 0,
        max_length: int = 2048,
        use_adaln: bool = True,
        num_classes: int = None,               # None = unconditional (no label embedding)
        label_dropout_prob: float = 0.0,        # CFG label dropout
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.dropout_prob = dropout_prob
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.layer_norm_before_last_layer = layer_norm_before_last_layer
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.ffn_bias = ffn_bias
        self.att_bias = att_bias
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.use_adaln = use_adaln
        self.num_classes = num_classes
        self.label_dropout_prob = label_dropout_prob


def modulate(x, shift, scale):
    """AdaLN modulation of a [B, L, D] sequence by per-sample shift and scale [B, D]."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                  Embedding layers for timesteps and class labels              #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps t into vectors: sinusoidal features followed by an MLP (as in DiT).

    NB: unlike the image experiments' counterpart, which relies on the sinusoidal default of
    10000, this honours config.max_period, and the released runs set it to 10
    (conf/model/35M.yaml). It changes the embedding, so the checkpoints depend on it.
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        self.frequency_embedding_size = config.frequency_embedding_size
        self.max_period = config.max_period
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """[N] (possibly fractional) timesteps -> [N, dim] sinusoidal features. From GLIDE."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size,
                                                max_period=self.max_period))


class LabelEmbedder(nn.Module):
    """Embeds class labels, with the label dropout that classifier-free guidance needs.
    Index num_classes is the unconditional (CFG) token."""

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_size = config.hidden_size
        self.label_dropout_prob = config.label_dropout_prob

        # +1 for the unconditional token
        self.embedding_table = nn.Embedding(config.num_classes + 1, config.hidden_size)

    def token_drop(self, labels, force_drop_ids=None):
        """Replaces a subset of the labels by the unconditional token."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.label_dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        # The trainer samples the dropout itself and passes force_drop_ids, so that validation
        # (train=False, force_drop_ids=None) reports the fully conditional loss.
        use_dropout = self.label_dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


#################################################################################
#                                  Encoder block                                #
#################################################################################

class EncoderBlock(nn.Module):
    """Transformer encoder block, conditioned on the noise level and class label.

    With use_adaln (the default) the conditioning vector c drives adaLN-Zero modulation of the
    attention and feed-forward sub-blocks, as in DiT. Otherwise the block is a plain pre-norm
    encoder block and c is instead added to the token embeddings in AMPLIFY.forward.
    """

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()

        self.config = config
        self.d_head = config.hidden_size // config.num_attention_heads

        # Attention
        self.q  = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.k  = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.v  = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.resid_dropout = nn.Dropout(config.dropout_prob)

        # Feedforward network
        act = config.hidden_act.lower()
        if act == "swiglu":
            # To keep the number of parameters and the amount of computation constant, we reduce the number of
            # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
            # avoid RuntimeError due to misaligned operand
            multiple_of = 8
            intermediate_size = int(2 * config.intermediate_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
            self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=config.ffn_bias)
        elif act == "relu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.ReLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        elif act == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        else:
            raise ValueError(f"Unsupported hidden_act: {config.hidden_act}")

        # LayerNorm rather than AMPLIFY's original RMSNorm, so that adaLN can scale and shift it
        self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)

        self.ffn_dropout = nn.Dropout(config.dropout_prob)

        if config.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        if self.config.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

            attn, contact = self._att_block(modulate(self.norm1(x), shift_msa, scale_msa), pad_mask, freqs_cis, output_attentions)
            x = x + gate_msa.unsqueeze(1) * attn

            x = x + gate_mlp.unsqueeze(1) * self._ff_block(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            # The conditioning vector was already added to x in AMPLIFY.forward
            attn, contact = self._att_block(self.norm1(x), pad_mask, freqs_cis, output_attentions)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))

        return x, contact

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # The attention weights are only materialised when they are asked for
        attn_weights = None
        if output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if pad_mask is not None:
                attn_weights = attn_weights + pad_mask
            attn_weights = attn_weights.softmax(-1)

        # Input and output are of dimension (B, H, M, K)
        attn = scaled_dot_product_attention(
            query=xq.transpose(1, 2),
            key=xk.transpose(1, 2),
            value=xv.transpose(1, 2),
            attn_mask=pad_mask,
            dropout_p=self.config.dropout_prob if self.training else 0,
        ).transpose(1, 2)

        attn_scores = self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.d_head))
        return (self.resid_dropout(attn_scores), attn_weights)

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


#################################################################################
#                                   Full model                                  #
#################################################################################

class AMPLIFYPreTrainedModel(PreTrainedModel):
    config_class = AMPLIFYConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)

        # adaLN-Zero: every conditioned block starts as the identity
        if hasattr(module, "adaLN_modulation") and isinstance(module.adaLN_modulation, nn.Sequential):
            last_layer = module.adaLN_modulation[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.constant_(last_layer.weight, 0)
            nn.init.constant_(last_layer.bias, 0)


class AMPLIFY(AMPLIFYPreTrainedModel):
    """Protein language model predicting per-token logits over the clean sequence.

    Args:
        config (AMPLIFYConfig): model configuration, usually built from the Hydra configuration.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs):
        super().__init__(config)

        self.config = config

        # NB: the order in which submodules are created fixes the random initialisation; do not reorder.
        self.timestep_encoder = TimestepEmbedder(config)

        self.label_embedder = LabelEmbedder(config) if config.num_classes is not None else None

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.layer_norm_after_embedding:
            self.layer_norm_1 = nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if config.layer_norm_before_last_layer:
            self.layer_norm_2 = nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, t, pad_mask=None, class_labels=None, force_drop_ids=None,
                output_hidden_states=False, output_attentions=False):
        """
        x: [B, L] token ids
        t: [B] noise levels in [0, 1]
        pad_mask: optional [B, L] ADDITIVE mask, 0 on real tokens and -inf on padding
        class_labels: [B] V-gene class labels (num_classes = unconditional)
        force_drop_ids: optional [B], 1 replaces the label with the unconditional token
        returns: MaskedLMOutput whose logits are [B, L, vocab_size]
        """
        hidden_states, attentions = [], []

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "AMPLIFY expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        self.freqs_cis = self.freqs_cis.to(x.device, non_blocking=True)
        freqs_cis = self.freqs_cis[: x.shape[1]]

        # Timestep and label embeddings combine into one conditioning vector c.
        # Named class_labels, not y, because y already means "denoising target tokens"
        # everywhere else in this codebase (data_collator.py, trainer.py).
        t = self.timestep_encoder(t)
        if self.label_embedder is not None and class_labels is not None:
            c = t + self.label_embedder(class_labels, self.training, force_drop_ids=force_drop_ids)
        else:
            c = t

        x = self.encoder(x)
        if not self.config.use_adaln:
            # Without adaLN, condition additively instead: add c to every token of the sequence
            x = x + c.unsqueeze(1)

        if self.config.layer_norm_after_embedding:
            x = self.layer_norm_1(x)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, c, pad_mask, freqs_cis, output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Classification head with layer norm
        logits = self.decoder(self.layer_norm_2(x) if self.config.layer_norm_before_last_layer else x)

        return MaskedLMOutput(logits=logits, hidden_states=hidden_states, attentions=attentions)
