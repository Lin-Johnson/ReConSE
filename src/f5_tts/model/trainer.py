from __future__ import annotations

import gc
import json
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


# trainer
class _TrainableParameterShadow(torch.nn.Module):
    """A lightweight module containing only the model's trainable parameters.

    The parameter names mirror the original model so ema_pytorch can update
    only ControlNet/Adapter parameters while the frozen F5-TTS backbone is
    excluded from the EMA copy.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.trainable_names = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            self.trainable_names.append(name)
            module = self
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                child = module._modules.get(part)
                if child is None:
                    child = torch.nn.Module()
                    module.add_module(part, child)
                module = child

            module.register_parameter(
                name_parts[-1],
                torch.nn.Parameter(parameter.detach().clone(), requires_grad=False),
            )



class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            ema_shadow = _TrainableParameterShadow(model)
            self.ema_model = EMA(
                model,
                ema_model=ema_shadow,
                include_online_model=False,
                **ema_kwargs,
            )
            self.ema_model.to(self.accelerator.device)

            ema_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            print(f"EMA tracks {ema_parameter_count:,} trainable parameters only")
            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate)
        else:
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = AdamW(trainable_params, lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            raw_model_state = {
                key: value.detach().cpu()
                for key, value in self.accelerator.unwrap_model(self.model).state_dict().items()
            }
            trainable_ema_state = {
                key: value.detach().cpu()
                for key, value in self.ema_model.state_dict().items()
            }

            # Keep the historical full EMA model key format so existing
            # inference code can still load EMA checkpoints. Frozen backbone
            # tensors are copied from the current model; only trainable
            # ControlNet/Adapter tensors come from the EMA shadow.
            raw_ema_state = {}
            for key, value in raw_model_state.items():
                ema_key = f"ema_model.{key}"
                raw_ema_state[ema_key] = trainable_ema_state.get(ema_key, value)
            for key, value in trainable_ema_state.items():
                if not key.startswith("ema_model."):
                    raw_ema_state[key] = value

            clean_model_state = {k: v for k, v in raw_model_state.items() if "qwen_encoder" not in k}
            clean_ema_state = {k: v for k, v in raw_ema_state.items() if "qwen_encoder" not in k}

            checkpoint = dict(
                model_state_dict=clean_model_state,
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=clean_ema_state,
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if getattr(train_dataset, "uses_qwen_encoder", False) and num_workers != 0:
            if self.is_main:
                print(
                    "Qwen feature extraction runs in the main process; "
                    "forcing num_workers=0 to avoid one encoder copy per DataLoader worker."
                )
            num_workers = 0

        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        # OOM debug: keep a short rolling trace of recent batches on each rank.
        # This adds only small CPU-side metadata and does not keep model tensors alive.
        recent_batch_debug = []
        oom_debug_keep = 12

        def _debug_value(value, max_items=64):
            if value is None:
                return None
            if torch.is_tensor(value):
                info = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                if value.numel() <= max_items:
                    try:
                        info["values"] = value.detach().cpu().tolist()
                    except Exception:
                        pass
                return info
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                if len(value) <= max_items:
                    return [_debug_value(v, max_items=max_items) for v in value]
                return {
                    "type": type(value).__name__,
                    "length": len(value),
                    "head": [_debug_value(v, max_items=max_items) for v in value[:8]],
                }
            if isinstance(value, dict):
                return {str(k): _debug_value(v, max_items=max_items) for k, v in value.items()}
            return str(value)

        def _cuda_mem_info():
            if not torch.cuda.is_available():
                return {}
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            }

        def _batch_debug_info(batch, epoch_idx, batch_idx, absolute_batch_idx, update):
            info = {
                "rank": self.accelerator.process_index,
                "local_rank": self.accelerator.local_process_index,
                "epoch": epoch_idx + 1,
                "batch_idx_in_current_dataloader": batch_idx,
                "absolute_batch_idx_in_epoch": absolute_batch_idx,
                "global_update_before_step": update,
                "batch_keys": list(batch.keys()),
                "cuda_before_forward": _cuda_mem_info(),
            }

            # Core tensors used by this trainer.
            for key in ("mel", "mel_lengths", "cond_mel", "qwen_feat", "durations", "text"):
                if key in batch:
                    info[key] = _debug_value(batch[key])

            # Capture likely sample-identification fields if the dataset/collate_fn provides them.
            id_keys = (
                "audio_path", "audio_paths", "path", "paths", "file", "files",
                "filename", "filenames", "id", "ids", "index", "indices",
                "sample_index", "sample_indices", "uid", "utt_id", "source", "dataset",
            )
            for key in id_keys:
                if key in batch and key not in info:
                    info[key] = _debug_value(batch[key])

            # Also keep lightweight non-tensor metadata not covered above.
            for key, value in batch.items():
                if key in info:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    info[key] = value
                elif isinstance(value, (list, tuple)) and len(value) <= 64:
                    if all(isinstance(v, (str, int, float, bool)) for v in value):
                        info[key] = list(value)

            return info

        def _save_oom_debug(info, error):
            debug_dir = os.path.join(self.checkpoint_path, "oom_debug")
            os.makedirs(debug_dir, exist_ok=True)

            rank = self.accelerator.process_index
            epoch_num = info.get("epoch", -1)
            abs_batch = info.get("absolute_batch_idx_in_epoch", -1)
            update = info.get("global_update_before_step", -1)

            info["error"] = str(error)
            info["cuda_at_oom"] = _cuda_mem_info()
            info["recent_batches"] = recent_batch_debug[-oom_debug_keep:]

            json_path = os.path.join(
                debug_dir,
                f"oom_rank{rank}_epoch{epoch_num}_batch{abs_batch}_update{update}.json",
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            # CUDA memory summary is often useful for distinguishing a true activation
            # spike from allocator fragmentation.
            if torch.cuda.is_available():
                summary_path = json_path.replace(".json", "_memory.txt")
                try:
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(torch.cuda.memory_summary())
                except Exception:
                    pass

            print(
                "\n" + "=" * 100
                + f"\nCUDA OOM captured on rank {rank}"
                + f"\nEpoch: {epoch_num}"
                + f"\nAbsolute batch index: {abs_batch}"
                + f"\nGlobal update: {update}"
                + f"\nDebug JSON: {json_path}"
                + "\n" + "=" * 100,
                flush=True,
            )

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch_idx, batch in enumerate(current_dataloader):
                # When resuming in the middle of the first resumed epoch,
                # enumerate(current_dataloader) restarts at 0. Recover the original
                # batch position in that epoch for easier reproduction.
                if exists(resumable_with_seed) and epoch == skipped_epoch:
                    absolute_batch_idx = skipped_batch + batch_idx
                else:
                    absolute_batch_idx = batch_idx

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                debug_info = _batch_debug_info(
                    batch=batch,
                    epoch_idx=epoch,
                    batch_idx=batch_idx,
                    absolute_batch_idx=absolute_batch_idx,
                    update=global_update,
                )
                recent_batch_debug.append(debug_info)
                if len(recent_batch_debug) > oom_debug_keep:
                    recent_batch_debug.pop(0)

                try:
                    with self.accelerator.accumulate(self.model):
                        text_inputs = batch["text"]
                        mel_spec = batch["mel"].permute(0, 2, 1)
                        mel_lengths = batch["mel_lengths"]
                        control_cond = batch["cond_mel"].permute(0, 2, 1)
                        qwen_feat = batch["qwen_feat"]
                        qwen_feat_mask = batch["qwen_feat_mask"]

                        # Record the exact post-collate shapes that enter the model.
                        debug_info["model_inputs"] = {
                            "mel_spec_shape": list(mel_spec.shape),
                            "mel_lengths": mel_lengths.detach().cpu().tolist(),
                            "control_cond_shape": list(control_cond.shape),
                            "qwen_feat": _debug_value(qwen_feat),
                            "qwen_feat_mask": _debug_value(qwen_feat_mask),
                            "text": _debug_value(text_inputs),
                        }

                        if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                            dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                            self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                        loss, cond, pred = self.model(
                            mel_spec,
                            text=text_inputs,
                            lens=mel_lengths,
                            noise_scheduler=self.noise_scheduler,
                            control_cond=control_cond,
                            qwen_feat=qwen_feat,
                            qwen_feat_mask=qwen_feat_mask,
                        )

                        debug_info["cuda_after_forward"] = _cuda_mem_info()

                        self.accelerator.backward(loss)

                        debug_info["cuda_after_backward"] = _cuda_mem_info()

                        if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        if self.is_main:
                            self.ema_model.update()

                        global_update += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                    if self.accelerator.is_local_main_process:
                        self.accelerator.log(
                            {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                        )
                        if self.logger == "tensorboard":
                            self.writer.add_scalar("loss", loss.item(), global_update)
                            self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                    if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update, last=True)

                    if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update)

                        if self.log_samples and self.accelerator.is_local_main_process:
                            ref_audio_len = mel_lengths[0].item()
                            cond_mel_spec = batch["cond_mel"][0].unsqueeze(0)

                            infer_text = [text_inputs[0]]
                            with torch.inference_mode():
                                generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    text=infer_text,
                                    duration=ref_audio_len,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                    control_cond=cond_mel_spec.permute(0, 2, 1),
                                    qwen_feat=qwen_feat[0].unsqueeze(0)
                                )
                                generated = generated.to(torch.float32)
                                gen_mel_spec = generated.permute(0, 2, 1).to(self.accelerator.device)
                                ref_mel_spec = batch["mel"][0].unsqueeze(0)

                                if self.vocoder_name == "vocos":
                                    gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                    ref_audio = vocoder.decode(ref_mel_spec).cpu()
                                    cond_audio = vocoder.decode(cond_mel_spec).cpu()
                                elif self.vocoder_name == "bigvgan":
                                    gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                    ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
                                    cond_audio = vocoder(cond_mel_spec).squeeze(0).cpu()

                            torchaudio.save(
                                f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                            )
                            torchaudio.save(
                                f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                            )
                            torchaudio.save(
                                f"{log_samples_path}/update_{global_update}_cond.wav", cond_audio, target_sample_rate
                            )
                            txt_path = f"{log_samples_path}/update_{global_update}_text.txt"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                content = infer_text[0]
                                if isinstance(content, list):
                                    content = "".join([str(x) for x in content])
                                f.write(content)
                            self.model.train()

                except torch.OutOfMemoryError as e:
                    # Do not swallow the exception in DDP: one rank skipping the batch
                    # while another rank continues can deadlock collective operations.
                    _save_oom_debug(debug_info, e)
                    raise

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
