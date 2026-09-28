"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.1,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        control_cond=None,
        qwen_feat=None,
        qwen_feat_mask=None,
    ):
        self.eval()
        # raw wave
        dtype = next(self.parameters()).dtype
        if cond.ndim == 2:
            cond = self.mel_spec(cond).permute(0, 2, 1)

        cond = cond.to(dtype)
        
        # Convert control_cond and align its dtype with the main input.
        if control_cond is not None:
            if control_cond.ndim == 2:
                control_cond = self.mel_spec(control_cond).permute(0, 2, 1)
            control_cond = control_cond.to(dtype)
        
        # Move Qwen features to the matching device and dtype.
        if qwen_feat is not None:
            qwen_feat = qwen_feat.to(device=cond.device, dtype=dtype)
        if qwen_feat_mask is not None:
            qwen_feat_mask = qwen_feat_mask.to(device=cond.device, dtype=torch.bool)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        if isinstance(text, torch.Tensor):
            text = text.long()

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        # Pad control_cond to match the conditioning sequence length.
        if control_cond is not None:
            control_cond_seq_len = control_cond.shape[1]
            control_cond = F.pad(control_cond, (0, 0, 0, max_duration - control_cond_seq_len), value=0.0)

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        ) 

        # Truncate control_cond and step_cond to the same sequence length.
        if control_cond is not None:
            min_len = min(cond.shape[1], control_cond.shape[1])
            step_cond = step_cond[:, :min_len, :]
            control_cond = control_cond[:, :min_len, :]

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  
            mask = None

        # Classifier-free guidance is not used by the ControlNet model. Keep
        # cfg_strength in the public signature for CLI/API compatibility.
        def fn(t, x):
            return self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
                cache=True,
                # Forward both control_cond and Qwen features.
                control_cond=control_cond,
                qwen_feat=qwen_feat,
                qwen_feat_mask=qwen_feat_mask,
            )
            
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss: 
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory


    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        # Preserve the optional control_cond input.
        control_cond=None,
        qwen_feat=None,
        qwen_feat_mask=None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        
        # Validate control_cond dimensions and move it to the model device.
        if control_cond is not None:
            if control_cond.ndim == 2:
                control_cond = self.mel_spec(control_cond).permute(0, 2, 1)
            control_cond = control_cond.to(device, dtype=dtype)

        if qwen_feat is not None:
            qwen_feat = qwen_feat.to(device=device, dtype=dtype)
        if qwen_feat_mask is not None:
            qwen_feat_mask = qwen_feat_mask.to(device=device, dtype=torch.bool)

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        if isinstance(text, torch.Tensor):
            text = text.long()

        if not exists(lens):  
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.ones((batch,), device=self.device).float()
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        x1 = inp
        x0 = torch.randn_like(x1)

        time = torch.rand((batch,), dtype=dtype, device=self.device)

        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        drop_audio_cond = random() < self.audio_drop_prob  
        if random() < self.cond_drop_prob:  
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=φ, 
            cond=cond, 
            text=text, 
            time=time, 
            drop_audio_cond=drop_audio_cond, 
            drop_text=drop_text, 
            mask=mask, 
            control_cond=control_cond,
            qwen_feat=qwen_feat,
            qwen_feat_mask=qwen_feat_mask,
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
