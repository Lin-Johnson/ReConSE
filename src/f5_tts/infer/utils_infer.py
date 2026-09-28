# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.model.control_f5 import ControlF5DiT

from transformers import WhisperFeatureExtractor


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

from f5_tts.model.qwen_encoder import Qwen3ASRAudioEncoder
PROJECT_ROOT = Path(__file__).resolve().parents[3]
QWAN_ASR_PATH = os.getenv(
    "F5TTS_QWEN_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3-ASR-1.7B"),
)
qwen_encoder = Qwen3ASRAudioEncoder.from_qwen3_asr_pretrained(
    QWAN_ASR_PATH,
    dtype=torch.float32,
    device="cuda",
    attn_implementation="eager"
)
qwen_encoder.eval()

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        print(f"Loading BigVGAN vocoder (is_local={is_local})...")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Select EMA or online weights explicitly. Older checkpoints may contain
    # only one of these state dictionaries.
    if isinstance(checkpoint, dict):
        if use_ema and "ema_model_state_dict" in checkpoint:
            raw_state_dict = checkpoint["ema_model_state_dict"]
            print("Loading EMA model weights for inference.")
        elif "model_state_dict" in checkpoint:
            raw_state_dict = checkpoint["model_state_dict"]
            print("Loading online model weights for inference.")
        else:
            raw_state_dict = checkpoint
    else:
        raw_state_dict = checkpoint

    new_state_dict = {}
    
    # Normalize namespace differences introduced by the ControlNet wrapper.
    for k, v in raw_state_dict.items():
        if k in {"initted", "step", "update"}:
            continue
        # Drop redundant Qwen parameters to save memory and avoid accidental overwrites.
        if "qwen_encoder" in k:
            continue
            
        new_key = k

        if new_key.startswith("ema_model."):
            new_key = new_key[len("ema_model.") :]
        
        # 1. Remove an optional base_model. prefix for compatibility.
        new_key = new_key.replace("base_model.", "")

        # 2. Handle the ControlF5DiT wrapper when the model expects
        # transformer.base_model.xxx but the checkpoint contains transformer.xxx.
        # Restore the base-model path in that case.
        if new_key.startswith("transformer."):
            # Keep adapters and control_mel_proj in the allowlist to avoid bad remapping.
            if not new_key.startswith("transformer.base_model.") and \
               not "controlnet" in new_key and \
               not "control_input_proj" in new_key and \
               not "control_mel_proj" in new_key and \
               not "adapters" in new_key and \
               not "qwen_connector" in new_key:
                new_key = new_key.replace("transformer.", "transformer.base_model.", 1)

        new_state_dict[new_key] = v

    # Use strict=False because ControlNet checkpoints may contain minor key differences.
    # This allows loading whenever the core weights match.
    info = model.load_state_dict(new_state_dict, strict=False)
    
    print("\n[Checkpoint Load Info]:")
    print(f"  - Missing keys: {len(info.missing_keys)}")
    print(f"  - Unexpected keys: {len(info.unexpected_keys)}")
    
    # Print a few missing keys when too many core keys are absent.
    if len(info.missing_keys) > 0:
        print(f"  - First 5 missing keys: {info.missing_keys[:5]}")
    if len(info.unexpected_keys) > 0:
        print(f"  - First 5 unexpected keys: {info.unexpected_keys[:5]}")

    del checkpoint
    del new_state_dict
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
    control_layers=0,
    adapter_layers=None,
    # Add the qwen_ckpt_path argument.
    qwen_ckpt_path=None,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)

    base_transformer = model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)
    control_layers = control_layers
    

    print(f"Injecting ControlNet with {control_layers} layers...")
    transformer = ControlF5DiT(
        base_transformer, 
        copy_blocks_num=control_layers,
        qwen_ckpt_path=qwen_ckpt_path,
        adapter_layers=adapter_layers,
    )
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

    # Load the original audio directly.
        aseg = AudioSegment.from_file(ref_audio_orig)
        
    # Do not split on silence; only enforce the length limit.
    # Truncate to the first 12 seconds when necessary.
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short.")
        
    # Optionally remove leading and trailing silence if desired.
        # aseg = remove_silence_edges(aseg)
        
    # Add 50 ms of trailing silence as a buffer.
        # aseg = aseg + AudioSegment.silent(duration=50)
        
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    control_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    qwen_fe=None,
):
    # Split the input text into batches
    audio, _ = torchaudio.load(ref_audio)
    control_audio, sr = torchaudio.load(control_audio)
    if gen_text == "":
        gen_text_batches = [""]
    else:
        max_chars = int(len(gen_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)

    # Prevent max_chars from becoming zero.
        max_chars = max(max_chars, 1)
        gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", repr(gen_text))

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            control_audio,
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            qwen_fe=qwen_fe,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    control_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    qwen_fe=None,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if control_audio.shape[0] > 1:
        control_audio = torch.mean(control_audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
        control_audio = control_audio * target_rms / rms
    
    # Record the original duration for later use.
    control_audio_sec = control_audio.shape[-1] / sr
    
    # --- 1. Prepare 24 kHz audio for the backbone. ---
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        # Resample control audio to 24 kHz before extracting the ControlNet mel.
        control_audio_24k = resampler(control_audio)
    else:
        control_audio_24k = control_audio
        
    audio = audio.to(device)
    control_audio_24k = control_audio_24k.to(device)
    
    # --- 2. Prepare 16 kHz features for Qwen. ---
    if sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(sr, 16000)
        control_audio_16k = resampler_16k(control_audio)
    else:
        control_audio_16k = control_audio
        
    qwen_feat = None
    qwen_feat_mask = None
    if qwen_fe is not None:
        control_audio_np = control_audio_16k.squeeze(0).cpu().numpy()
        qwen_feats = qwen_fe(
            control_audio_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            return_attention_mask=True
        )
        # ==============================================================
        # Keep preprocessing identical to training.
        # Preserve the full 3000 frames (30 seconds) extracted by Whisper.
        input_features = qwen_feats["input_features"].to(device)  # shape: (1, 128, 3000)
    
        feature_lens = qwen_feats["attention_mask"].sum(dim=-1).to(device)

        with torch.no_grad():
            qwen_out = qwen_encoder(
                input_features, 
                feature_lens=feature_lens, 
                output_hidden_states=True
            )
        
        qwen_feat = qwen_out.hidden_states[18]
        qwen_feat_mask = torch.ones(
            qwen_feat.shape[:2], device=qwen_feat.device, dtype=torch.bool
        )
    
    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = math.ceil(audio.shape[-1] / hop_length)
                # Convert duration to the 24 kHz frame count instead of using the resampled shape.
        control_audio_len = math.ceil(control_audio_sec * target_sample_rate / hop_length)
        
        if fix_duration is not None:
            # duration = int(fix_duration * target_sample_rate / hop_length)
            duration = control_audio_len
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            # duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
            duration = control_audio_len

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                # Restore ControlNet input and pass both control_cond and Qwen features.
                control_cond=control_audio_24k,
                qwen_feat=qwen_feat,
                qwen_feat_mask=qwen_feat_mask,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            # generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
