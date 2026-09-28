"""
Standalone Qwen3-ASR audio encoder.

This file is an excerpt of `qwen_asr.core.transformers_backend.modeling_qwen3_asr`
that contains *only* the audio tower (and its config). It can be used in any
environment that has `transformers >= 4.49` (it does not require any of the
4.55+ symbols pulled in by the text decoder, e.g. `TransformersKwargs` or
`check_model_inputs`).

How it loads weights:
  - `Qwen3ASRAudioEncoder.from_qwen3_asr_pretrained(model_path)` opens the
    `model.safetensors.index.json` of a Qwen3-ASR checkpoint, finds the keys
    prefixed with `thinker.audio_tower.`, and loads them into the encoder.
  - It also picks up `audio_config` from the `config.json`'s `thinker_config`
    block.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    """Mirror of `qwen_asr...configuration_qwen3_asr.Qwen3ASRAudioEncoderConfig`."""

    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


# --------------------------------------------------------------------------------------
# Helpers (verbatim from modeling_qwen3_asr.py)
# --------------------------------------------------------------------------------------


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# --------------------------------------------------------------------------------------
# Model components
# --------------------------------------------------------------------------------------


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


# --------------------------------------------------------------------------------------
# Encoder + loading helpers
# --------------------------------------------------------------------------------------


class Qwen3ASRAudioEncoder(PreTrainedModel):
    """Audio tower of Qwen3-ASR-1.7B (verbatim forward, simplified base class)."""

    config_class = Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            [Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        aftercnn_lens: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Forward identical to qwen_asr's `Qwen3ASRAudioEncoder.forward`.

        Args:
            input_features: 2-D tensor of shape (num_mel_bins, T_mel) for one
                audio sample (the same per-sample call pattern that
                `Qwen3ASRThinkerForConditionalGeneration.get_audio_features` uses).
            feature_lens: 1-element long tensor with the unpadded mel length.
            output_hidden_states: if True, also return per-layer activations
                in `output.hidden_states` (length = encoder_layers + 2):
                  [0]      conv stack output + positional embedding
                           (= input to transformer layer 1)
                  [1..N]   output of each transformer encoder layer (N=24)
                  [N+1]    output of `ln_post` (final pre-projection state)
                The `last_hidden_state` is the post-projection 2048-d output.
        """
        
        # Flatten 3D batched input into the contiguous 2D format expected by Qwen.
        was_3d = False
        if input_features.ndim == 3:
            was_3d = True
            batch_size = input_features.size(0)
            
            # Ensure feature_lens is strictly 1D inside the interceptor.
            if feature_lens is not None:
                feature_lens = feature_lens.view(-1)
                
            feat_list = []
            for i in range(batch_size):
                # Remove zero padding, slice each sequence to feature_lens, and concatenate.
                feat_list.append(input_features[i, :, :feature_lens[i]])
            input_features = torch.cat(feat_list, dim=1)  # Result: [128, sum(T)].

        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(
            1, 2
        )
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )

        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for encoder_layer in self.layers:
            (hidden_states,) = encoder_layer(hidden_states, cu_seqlens)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.ln_post(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        
        # Unflatten continuous Qwen output back to a padded 3D tensor for 3D input.
        if was_3d:
            out_hidden_states = []
            start = 0
            for length in aftercnn_lens:
                length_int = int(length.item())
                end = start + length_int
                # Slice each utterance to its actual post-CNN length.
                out_hidden_states.append(hidden_states[start:end, :])
                start = end
            # Repad variable-length sequences to [Batch, max_T_native, Dim].
            hidden_states = nn.utils.rnn.pad_sequence(out_hidden_states, batch_first=True, padding_value=0.0)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
        
    # ----- Loading utilities -----

    @classmethod
    def from_qwen3_asr_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        attn_implementation: str = "sdpa",
    ) -> "Qwen3ASRAudioEncoder":
        """Build the audio encoder from a Qwen3-ASR-1.7B checkpoint directory.

        Reads only the `thinker.audio_tower.*` tensors out of the safetensors
        shards (so the text decoder / lm_head / embeddings are never loaded).
        """
        with open(os.path.join(model_path, "config.json"), "r") as f:
            full_cfg = json.load(f)
        audio_cfg_dict = full_cfg["thinker_config"]["audio_config"]
        config = Qwen3ASRAudioEncoderConfig(**audio_cfg_dict)
        config._attn_implementation = attn_implementation

        model = cls(config)

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        single_path = os.path.join(model_path, "model.safetensors")

        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            files_to_keys: dict[str, list[str]] = {}
            for full_key, fname in weight_map.items():
                if not full_key.startswith("thinker.audio_tower."):
                    continue
                files_to_keys.setdefault(fname, []).append(full_key)
            file_iter = files_to_keys.items()
        elif os.path.exists(single_path):
            file_iter = [(os.path.basename(single_path), None)]
        else:
            raise FileNotFoundError(
                f"No safetensors index or single safetensors file in {model_path}"
            )

        from safetensors import safe_open

        state_dict: dict[str, torch.Tensor] = {}
        prefix = "thinker.audio_tower."
        for fname, keys in file_iter:
            with safe_open(os.path.join(model_path, fname), framework="pt") as f:
                if keys is None:
                    keys = [k for k in f.keys() if k.startswith(prefix)]
                for k in keys:
                    short_key = k[len(prefix):]
                    state_dict[short_key] = f.get_tensor(k)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # `positional_embedding.positional_embedding` is a non-persistent buffer
        # so it intentionally never appears in the checkpoint.
        missing = [k for k in missing if k != "positional_embedding.positional_embedding"]
        if missing:
            raise RuntimeError(f"Missing keys when loading audio encoder: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading audio encoder: {unexpected}")

        model = model.to(device=device, dtype=dtype).eval()
        return model


if __name__ == "__main__":
    import numpy as np
    from transformers import WhisperFeatureExtractor

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ckpt = os.getenv("F5TTS_QWEN_PATH", "models/Qwen3-ASR-1.7B")
    encoder = Qwen3ASRAudioEncoder.from_qwen3_asr_pretrained(
        ckpt, dtype=dtype, device=device, attn_implementation="eager"
    )

    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)                # (32000,) 2s noise

    fe = WhisperFeatureExtractor.from_pretrained(ckpt)
    feats = fe(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    feature_lens = feats["attention_mask"].sum(dim=-1).to(device)     # (1,) -> tensor([200])
    feat_len = int(feature_lens.item())
    input_features = (
        feats["input_features"][0, :, :feat_len].to(device=device, dtype=dtype)
    )                                                                  # (128, 200)

    with torch.no_grad():
        out = encoder(input_features, feature_lens=feature_lens, output_hidden_states=True)

    print(f"audio.shape             = {audio.shape}")                 # (32000,)
    print(f"input_features.shape    = {tuple(input_features.shape)}") # (128, 200)
    print(f"feature_lens            = {feature_lens.tolist()}")        # [200]
    print(f"d_model                 = {encoder.config.d_model}")       # 1024
    print(f"last_hidden_state.shape = {tuple(out.last_hidden_state.shape)}")  # (T_native, 2048)
    print(f"hidden_states[18].shape = {tuple(out.hidden_states[18].shape)}")  # (T_native, 1024)
