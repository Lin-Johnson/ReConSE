from copy import deepcopy

import torch
from torch import nn

from f5_tts.model.modules import DiTBlock


class IPAdapter(nn.Module):
    def __init__(self, qwen_feat_dim=1024, ip_num_heads=16):
        super().__init__()
        self.num_heads = ip_num_heads

        # Use the native Qwen feature dimension for K/V projection without reduction.
        self.to_k_ip = nn.Linear(qwen_feat_dim, qwen_feat_dim, bias=False)
        self.to_v_ip = nn.Linear(qwen_feat_dim, qwen_feat_dim, bias=False)

        # Zero-initialize so the pretrained model is unchanged at startup.
        nn.init.zeros_(self.to_k_ip.weight)
        nn.init.zeros_(self.to_v_ip.weight)

    def forward(self, ip_feat):
        if ip_feat.dim() == 2:
            ip_feat = ip_feat.unsqueeze(0)  # Add a batch dimension: [1, T, 1024].
        B, N, D = ip_feat.shape
        H = self.num_heads
        head_dim = D // H

        # Project IP features.
        k_ip = self.to_k_ip(ip_feat).view(B, N, H, head_dim).transpose(1, 2)
        v_ip = self.to_v_ip(ip_feat).view(B, N, H, head_dim).transpose(1, 2)

        return k_ip, v_ip


class ControlDiTBlockHalf(nn.Module):
    def __init__(self, base_block: DiTBlock, blocks_num: int, dim: int):
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.blocks_num = blocks_num

        self.before_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.before_proj.weight)
        nn.init.zeros_(self.before_proj.bias)
        
        self.after_proj = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(blocks_num)]
        )

        for proj in self.after_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x, t, block_index, mask=None, rope=None, c=None):

        if block_index == 0:
            # First shared control block.
            c = self.before_proj(c)
            c = self.copied_block(c + x, t, mask=mask, rope=rope)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c + x, t, mask=mask, rope=rope)

        c_skip = self.after_proj[block_index](c)
        
        return c, c_skip


class ControlF5DiT(nn.Module):
    def __init__(
        self,
        base_model,
        copy_blocks_num: int = 4,
        qwen_ckpt_path=None,
        adapter_layers=None,
    ):
        super().__init__()
        self.base_model = base_model
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.transformer_blocks)

        if self.copy_blocks_num < 1:
            raise ValueError("copy_blocks_num must be at least 1")

        if self.copy_blocks_num >= self.total_blocks_num:
            raise ValueError(
                f"copy_blocks_num={self.copy_blocks_num} must be smaller than "
                f"total_blocks_num={self.total_blocks_num}"
            )
        
        inner_dim = base_model.dim
        if adapter_layers is None:
            adapter_layers = [0, 2, 4, 10, 11, 12, 13, 15, 16, 21]

        self.controlnet = ControlDiTBlockHalf(
            base_model.transformer_blocks[0], copy_blocks_num, inner_dim
        )

        num_heads = base_model.transformer_blocks[0].attn.heads
        # Attach adapters to the selected backbone layers.
        self.adapters = nn.ModuleDict(
            {
                str(i): IPAdapter(qwen_feat_dim=1024, ip_num_heads=num_heads)
                for i in adapter_layers
            }
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _get_ip_features(self, index, qwen_hidden):
        if qwen_hidden is not None and str(index) in self.adapters:
            return self.adapters[str(index)](qwen_hidden)
        return None, None

    def forward(
        self,
        x,
        cond,
        text,
        time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cache=False,
        control_cond=None,  # mel_spec [B, N, 80]
        qwen_feat=None,
        qwen_feat_mask=None,
        **kwargs,
    ):
        # --- A. Embedding ---
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t_emb = self.base_model.time_embed(time)

        control_scale = 1

        # --- B. Controlnet Input ---
        current_c = None
        c = control_cond

        # Get the main-path input features (the noised phi).
        x_main = self.base_model.get_input_embed(x, cond, text, True, False, cache, mask)

        if c is not None:
            current_c = self.base_model.get_input_embed(c, cond, text, True, True, cache, mask)

        rope = self.base_model.rotary_embed.forward_from_seq_len(seq_len)

        if qwen_feat is not None and qwen_feat_mask is None:
            qwen_feat_mask = torch.ones(
                qwen_feat.shape[:2], device=qwen_feat.device, dtype=torch.bool
            )

        # --- C. Post-injection logic ---
        x = x_main

        # Run the first backbone block.
        k_ip, v_ip = self._get_ip_features(0, qwen_feat)

        if self.base_model.checkpoint_activations:
            x = torch.utils.checkpoint.checkpoint(
                self.base_model.transformer_blocks[0], x, t_emb, mask, rope,
                k_ip, v_ip, qwen_feat_mask, use_reentrant=False
            )
        else:
            x = self.base_model.transformer_blocks[0](
                x,
                t_emb,
                mask=mask,
                rope=rope,
                ip_k=k_ip,
                ip_v=v_ip,
                ip_mask=qwen_feat_mask,
            )

        if c is not None:
            # Update x and c.
            for index in range(1, self.copy_blocks_num + 1):

                k_ip, v_ip = self._get_ip_features(index, qwen_feat)
                block_index = index - 1

                if self.base_model.checkpoint_activations:

                    def controlnet_forward(
                        x_, t_emb_, mask_, rope_, current_c_, block_index_=block_index
                    ):
                        return self.controlnet(
                            x_, t_emb_, block_index_, mask=mask_, rope=rope_, c=current_c_
                        )

                    current_c, c_skip = torch.utils.checkpoint.checkpoint(
                        controlnet_forward, x, t_emb, mask, rope, current_c,
                        use_reentrant=False
                    )
                else:
                    current_c, c_skip = self.controlnet(
                        x, t_emb, block_index, mask=mask, rope=rope, c=current_c
                    )

                x_main_in = x + control_scale * c_skip

                if self.base_model.checkpoint_activations:
                    x = torch.utils.checkpoint.checkpoint(
                        self.base_model.transformer_blocks[index], x_main_in, t_emb,
                        mask, rope, k_ip, v_ip, qwen_feat_mask, use_reentrant=False
                    )
                else:
                    x = self.base_model.transformer_blocks[index](
                        x_main_in,
                        t_emb,
                        mask=mask,
                        rope=rope,
                        ip_k=k_ip,
                        ip_v=v_ip,
                        ip_mask=qwen_feat_mask,
                    )

            # Update x.
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                k_ip, v_ip = self._get_ip_features(index, qwen_feat)
                x_main_in = x

                if self.base_model.checkpoint_activations:
                    x = torch.utils.checkpoint.checkpoint(
                        self.base_model.transformer_blocks[index], x_main_in, t_emb,
                        mask, rope, k_ip, v_ip, qwen_feat_mask, use_reentrant=False
                    )
                else:
                    x = self.base_model.transformer_blocks[index](
                        x_main_in,
                        t_emb,
                        mask=mask,
                        rope=rope,
                        ip_k=k_ip,
                        ip_v=v_ip,
                        ip_mask=qwen_feat_mask,
                    )

        else:
            for index in range(1, self.total_blocks_num):
                k_ip, v_ip = self._get_ip_features(index, qwen_feat)

                if self.base_model.checkpoint_activations:
                    x = torch.utils.checkpoint.checkpoint(
                        self.base_model.transformer_blocks[index], x, t_emb, mask, rope,
                        k_ip, v_ip, qwen_feat_mask, use_reentrant=False
                    )
                else:
                    x = self.base_model.transformer_blocks[index](
                        x,
                        t_emb,
                        mask=mask,
                        rope=rope,
                        ip_k=k_ip,
                        ip_v=v_ip,
                        ip_mask=qwen_feat_mask,
                    )

        # --- D. Post-processing ---
        x = self.base_model.norm_out(x, t_emb)
        output = self.base_model.proj_out(x)

        return output
