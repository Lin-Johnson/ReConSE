import json
import os
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QWAN_ASR_PATH = os.getenv(
    "F5TTS_QWEN_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3-ASR-1.7B"),
)

from f5_tts.model.qwen_encoder import Qwen3ASRAudioEncoder
from transformers import WhisperFeatureExtractor

_qwen_encoder = None
_feature_extractor = None


def _get_qwen_components(qwen_ckpt_path=QWAN_ASR_PATH):
    global _qwen_encoder, _feature_extractor
    if _qwen_encoder is None:
        _qwen_encoder = Qwen3ASRAudioEncoder.from_qwen3_asr_pretrained(
            qwen_ckpt_path,
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
        )
        _qwen_encoder.eval()
        for param in _qwen_encoder.parameters():
            param.requires_grad = False
    if _feature_extractor is None:
        _feature_extractor = WhisperFeatureExtractor.from_pretrained(qwen_ckpt_path)
    return _qwen_encoder, _feature_extractor


def extract_qwen_feat(audio_16k, qwen_ckpt_path=QWAN_ASR_PATH):
    qwen_encoder, feature_extractor = _get_qwen_components(qwen_ckpt_path)

    device = audio_16k.device

    # 1. Convert to NumPy for the Whisper feature extractor.
    audio_16k_np = audio_16k.squeeze().cpu().numpy()
    
    # 2. Extract features.
    feats = feature_extractor(
        audio_16k_np, 
        sampling_rate=16000, 
        return_tensors="pt", 
        return_attention_mask=True
    )
    
    # 3. Retrieve the features and move them to the target device.
    input_features = feats["input_features"].to(device)  # shape: (1, 128, 3000)
    
    # 4. Derive the valid feature lengths from attention_mask.
    feature_lens = feats["attention_mask"].sum(dim=-1).to(device)

    # 5. Encode the features with Qwen.
    with torch.no_grad():
        qwen_out = qwen_encoder(
            input_features, 
            feature_lens=feature_lens, 
            output_hidden_states=True
        )
    
    # print(f"audio_16k shape: {audio_16k.shape}, extracted feature shape: {feats['input_features'].shape}, features_len: {feature_lens}, qwen_out hidden states shape: {qwen_out.hidden_states[18].shape}")  # Debug log
    

    return qwen_out.hidden_states[18]  # [1000, D]


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        # Accept the Qwen checkpoint path.
        qwen_ckpt_path: str | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.qwen_ckpt_path = qwen_ckpt_path or QWAN_ASR_PATH
        self.uses_qwen_encoder = not preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = row["duration"]
        cond_audio_path = row["cond_audio_path"]

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
            cond_mel_spec = torch.tensor(row.get("cond_mel_spec", []))
            qwen_feat = torch.tensor(row["qwen_feat"])

        else:
            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != self.target_sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
            mel_spec = self.mel_spectrogram(audio).squeeze(0)

            # --- 2. Process the noisy control audio. ---
            cond_audio, cond_sr = torchaudio.load(cond_audio_path)
            if cond_audio.shape[0] > 1:
                cond_audio = torch.mean(cond_audio, dim=0, keepdim=True)
                
            # Extract the 24 kHz conditioning mel for validation visualization and logging.
            if cond_sr != self.target_sample_rate:
                resampler_24k = torchaudio.transforms.Resample(cond_sr, self.target_sample_rate)
                cond_audio_24k = resampler_24k(cond_audio)
            else:
                cond_audio_24k = cond_audio
            cond_mel_spec = self.mel_spectrogram(cond_audio_24k)  # [1, 100, T]
            cond_mel_spec = cond_mel_spec.squeeze(0)  

            if cond_sr != 16000:
                resampler_16k = torchaudio.transforms.Resample(cond_sr, 16000)
                cond_audio_16k = resampler_16k(cond_audio)
            else:
                cond_audio_16k = cond_audio

                
            qwen_feat = extract_qwen_feat(cond_audio_16k, self.qwen_ckpt_path)

        return {
            "mel_spec": mel_spec,
            "cond_mel_spec": cond_mel_spec,
            "qwen_feat": qwen_feat,
            "text": text,
        }



# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            frame_len = data_source.get_frame_len(idx)

            # Skip samples shorter than 0.3 s or longer than 30 s.
            min_frames = 0.3 * data_source.target_sample_rate / data_source.hop_length
            max_frames = 30.0 * data_source.target_sample_rate / data_source.hop_length

            if frame_len < min_frames or frame_len > max_frames:
                continue

            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset

def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    # Expose qwen_ckpt_path in the dataset loader interface.
    qwen_ckpt_path: str | None = None,
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            # Pass qwen_ckpt_path to the dataset.
            qwen_ckpt_path=qwen_ckpt_path,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        preprocessed_mel = False
        train_dataset = CustomDataset(
            train_dataset, 
            durations=durations, 
            preprocessed_mel=preprocessed_mel, 
            # Pass qwen_ckpt_path to the dataset.
            qwen_ckpt_path=qwen_ckpt_path,
            **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation

def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    cond_mel_specs = [item.get("cond_mel_spec", item["mel_spec"]).squeeze(0) for item in batch]

    def pad_spec(specs, max_len):
        padded = []
        for spec in specs:
            pad = (0, max_len - spec.size(-1))
            padded.append(F.pad(spec, pad, value=0))
        return torch.stack(padded)

    padded_mel_specs = pad_spec(mel_specs, max_mel_length)
    padded_cond_mel_specs = pad_spec(cond_mel_specs, max_mel_length)

    adapter_feats = [item["qwen_feat"] for item in batch] 
    
    adapter_feats = [feat.squeeze(0) if feat.dim() == 3 else feat for feat in adapter_feats]
    
    adapter_lengths = torch.LongTensor([feat.shape[0] for feat in adapter_feats])
    max_adapter_length = adapter_lengths.amax()

    def pad_adapter_feat(feats, max_len):
        padded = []
        for feat in feats:
            pad_amount = max_len - feat.size(0)
            pad = (0, 0, 0, pad_amount) 
            padded.append(F.pad(feat, pad, value=0))
        return torch.stack(padded)

    padded_adapter_feats = pad_adapter_feat(adapter_feats, max_adapter_length)
    qwen_feat_mask = (
        torch.arange(max_adapter_length)[None, :] < adapter_lengths[:, None]
    )

    text = [item["text"] for item in batch]
    
    return {
        "mel": padded_mel_specs,
        "cond_mel": padded_cond_mel_specs,
        "mel_lengths": mel_lengths,
        "qwen_feat": padded_adapter_feats, 
        "qwen_feat_mask": qwen_feat_mask,
        "text": text,
    }
