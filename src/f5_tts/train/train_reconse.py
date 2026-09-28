# accelerate launch src/f5_tts/train/train_reconse.py --config-name ReConSE

import os
import torch
from importlib.resources import files
from pathlib import Path
import hydra
import safetensors.torch
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.control_f5 import ControlF5DiT 

from f5_tts.infer.utils_infer import load_model

def load_pretrained_dit(checkpoint_path, model_cls, model_arc, vocab_size, mel_dim, device="cpu"):
    """Load a pretrained DiT model."""
    print(f"Loading pretrained DiT from {checkpoint_path}")
    
    # Convert model_arc from an OmegaConf object to a regular dictionary.
    if hasattr(model_arc, '_items'):
        model_arc_dict = OmegaConf.to_container(model_arc, resolve=True)
    else:
        model_arc_dict = model_arc
    
    # Create the model.
    model = model_cls(
        **model_arc_dict,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    )
    
    # Load the checkpoint.
    if checkpoint_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    print("Checkpoint keys:")
    for k in state_dict.keys():
        print(f"  - {k}")
    
    # Unwrap nested state_dict entries.
    if 'ema_model_state_dict' in state_dict:
        print("Found ema_model_state_dict; extracting nested weights...")
        state_dict = state_dict['ema_model_state_dict']
    elif 'model_state_dict' in state_dict:
        print("Found model_state_dict; extracting nested weights...")
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        print("Found state_dict; extracting nested weights...")
        state_dict = state_dict['state_dict']
    
    # Extract transformer weights from the EMA model.
    transformer_state_dict = {}
    
    # The checkpoint may contain several key formats.
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            # Use transformer weights directly.
            new_key = k.replace("transformer.", "")
            transformer_state_dict[new_key] = v
        elif k.startswith("ema_model.transformer."):
            # Remove the prefix.
            new_key = k.replace("ema_model.transformer.", "")
            transformer_state_dict[new_key] = v
        elif k.startswith("model.transformer."):
            # Fall back to regular model weights when no EMA model is present.
            new_key = k.replace("model.transformer.", "")
            transformer_state_dict[new_key] = v
    
    # If no transformer weights were found, try a more permissive match.
    if len(transformer_state_dict) == 0:
        print("Trying a permissive transformer-weight match...")
        for k, v in state_dict.items():
            if "transformer" in k and "input_embed" not in k and "time_embed" not in k:
                # Clean up the key name.
                if k.startswith("ema_model."):
                    k = k.replace("ema_model.", "")
                if k.startswith("model."):
                    k = k.replace("model.", "")
                # Remove possible nested prefixes.
                if k.startswith("transformer."):
                    k = k.replace("transformer.", "")
                transformer_state_dict[k] = v
    
    print(f"Extracted {len(transformer_state_dict)} transformer weights")
    
    if len(transformer_state_dict) == 0:
        print("Warning: transformer weights not found; loading the full state_dict...")
        # Let PyTorch match keys from the full state_dict.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    else:
        # Load weights.
        missing_keys, unexpected_keys = model.load_state_dict(transformer_state_dict, strict=False)
    
    print("Weight loading result:")
    print(f"  - Missing keys: {len(missing_keys)}")
    print(f"  - Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print("First 10 missing keys:")
        for k in missing_keys[:10]:
            print(f"  - {k}")
    
    if unexpected_keys:
        print("First 10 unexpected keys:")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")
    
    return model.to(device)

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="ReConSE")
def main(model_cfg):

    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    print(f"\n[STEP 1] Loading pretrained backbone model...")

    project_root = Path(__file__).resolve().parents[3]
    default_qwen_path = str(project_root / "models" / "Qwen3-ASR-1.7B")
    default_base_ckpt = str(project_root / "models" / "F5TTS_v1_Base" / "model_1250000.safetensors")
    qwen_ckpt_path = os.getenv("F5TTS_QWEN_PATH", default_qwen_path)
    ckpt_file = os.getenv("F5TTS_BASE_CKPT", default_base_ckpt)
    ckpt_file = str(getattr(model_cfg.model, "pretrained_checkpoint", ckpt_file))
    qwen_ckpt_path = str(getattr(model_cfg.model, "qwen_ckpt_path", qwen_ckpt_path))

    device = "cpu"  # Let Accelerate place each process on its local GPU
    pretrained_dit = load_pretrained_dit(
        checkpoint_path=ckpt_file,
        model_cls=model_cls,
        model_arc=model_arc,
        vocab_size=vocab_size,
        mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        device=device
    )

    # Create ControlNet.
    print(f"\n[STEP 2] Creating ControlNet...")
    from f5_tts.model.control_f5 import ControlF5DiT
    
    control_layers = getattr(model_cfg.model, "control_layers", 4)
    adapter_layers = list(
        getattr(model_cfg.model, "adapter_layers", [0, 2, 4, 10, 11, 12, 13, 15, 16, 21])
    )
    control_transformer = ControlF5DiT(pretrained_dit, copy_blocks_num=control_layers,
                            qwen_ckpt_path=qwen_ckpt_path,
                            adapter_layers=adapter_layers)
    
    # Freeze backbone parameters.
    for param in control_transformer.base_model.parameters():
        param.requires_grad = False
    
    # Count parameters.
    trainable_params = sum(p.numel() for p in control_transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in control_transformer.parameters())
    print(f"\nControlNet parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    # Print trainable parameters.
    print("\nTrainable parameters:")
    for name, param in control_transformer.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name}: {param.numel():,}")
    
    # set model
    model = CFM(
        transformer=control_transformer,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    model.to(torch.float32)

    # # Freezing Backbone
    # print(f"\n[STEP 3] Freezing Backbone...")
    # for param in model.parameters():
    #     param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if "controlnet" in name or "control_input_proj" in name:
    #         param.requires_grad = True
    #         print(f"✅ Trainable: {name}")
    #     else:
    #         print(f"❌ Frozen:   {name}")

    # Initialize Trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}_controlnet")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="F5-TTS-ControlNet",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    # Load Dataset
    train_dataset = load_dataset(model_cfg.datasets.name, model_cfg.model.tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec,
                                qwen_ckpt_path=qwen_ckpt_path)
    
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining started. Trainable Params (Optimizer size): {trainable_params_count / 1e6:.2f}M")

    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )

if __name__ == "__main__":
    main()
