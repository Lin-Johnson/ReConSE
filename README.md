# ReConSE: Recurrent ControlNet Speech Enhancement

<p align="center">
  <a href="https://lin-johnson.github.io/ReConSE-Demo/"><img src="https://img.shields.io/badge/Project-Demo-2563eb?style=flat-square" alt="Project Demo"></a>
</p>

ReConSE is a generative speech-enhancement system built on the frozen F5-TTS
flow-matching DiT backbone. It combines a shared recurrent ControlNet with
Qwen3-ASR audio representations to enhance degraded speech with or without a
target transcript.

## Demo

Listen to examples and view results on the [ReConSE Demo Page](https://lin-johnson.github.io/ReConSE-Demo/).

## Installation

Create a Python 3.11 environment and install a CUDA-compatible PyTorch and
torchaudio pair first. For CUDA 12.8, for example:

```bash
conda create -n reconse python=3.11 -y
conda activate reconse
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt
pip install -e . --no-deps
```

## External assets

The repository does not include model checkpoints or training data. Configure
the paths to the F5-TTS base model, Qwen3-ASR encoder, Vocos checkpoint, and
vocabulary in `src/f5_tts/configs/ReConSE.yaml`, under `assets`. Environment
variables override the repository-relative defaults:

```bash
export F5TTS_BASE_CKPT=/path/to/F5TTS_v1_Base/model_1250000.safetensors
export F5TTS_QWEN_PATH=/path/to/Qwen3-ASR-1.7B
export F5TTS_VOCOS_PATH=/path/to/vocos-mel-24khz
export F5TTS_VOCAB_PATH=/path/to/vocab.txt
```

The complete ReConSE checkpoint is deliberately selected explicitly with
`--ckpt_file` at inference time.

## Training

```bash
accelerate launch src/f5_tts/train/train_reconse.py --config-name ReConSE
```

Adjust the dataset, optimization, ControlNet, and adapter settings in
`src/f5_tts/configs/ReConSE.yaml`.

## Inference

The repository includes one DNS Challenge example:

```text
examples/0.wav
examples/0.txt
```

### Without a target transcript

Omit `--text_dir`; the accompanying text is used only as the reference
transcript.

```bash
python src/f5_tts/infer/infer_dir.py \
  --model ReConSE \
  --model_cfg src/f5_tts/configs/ReConSE.yaml \
  --ckpt_file /path/to/reconse_model.pt \
  --ref_audio examples/0.wav \
  --ref_text "$(cat examples/0.txt)" \
  --control_audio_dir examples \
  --output_dir outputs/example_without_text
```

### With a target transcript

Pass the directory containing text files named after the input audio files.

```bash
python src/f5_tts/infer/infer_dir.py \
  --model ReConSE \
  --model_cfg src/f5_tts/configs/ReConSE.yaml \
  --ckpt_file /path/to/reconse_model.pt \
  --ref_audio examples/0.wav \
  --ref_text "$(cat examples/0.txt)" \
  --control_audio_dir examples \
  --text_dir examples \
  --output_dir outputs/example_with_text
```

## Acknowledgements

This project builds on [F5-TTS](https://github.com/SWivid/F5-TTS) and uses
Qwen3-ASR and Vocos as external components.

## License

This code is released under the [MIT License](LICENSE).
