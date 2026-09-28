# F5-TTS ControlNet Speech Enhancement

This repository contains a speech-enhancement system built on the F5-TTS
flow-matching DiT backbone. The project adds a shared recurrent ControlNet and
a Qwen3-ASR audio representation adapter for enhancement from degraded speech.

The project-specific workflow uses:

- Vocos mel extraction and waveform synthesis;
- a frozen pretrained F5-TTS DiT backbone;
- trainable ControlNet blocks;
- Qwen3-ASR audio features as conditioning;
- Hydra configuration and Accelerate for multi-GPU training.

Large model checkpoints and datasets are intentionally not included.

## Installation

Create a clean environment, install a PyTorch and torchaudio pair matching
the target CUDA version, then install the project dependencies.

    conda create -n f5-tts-controlnet python=3.11 -y
    conda activate f5-tts-controlnet

For example, for CUDA 12.8:

    pip install torch==2.8.0 torchaudio==2.8.0       --index-url https://download.pytorch.org/whl/cu128

Install FFmpeg and the Python dependencies:

    conda install -c conda-forge ffmpeg -y
    pip install -r requirements.txt
    pip install -e . --no-deps

For other CUDA, ROCm, XPU, or CPU builds, install the corresponding PyTorch
wheels first.

## External models and paths

Set local paths with environment variables before training or inference:

    export F5TTS_QWEN_PATH="$PWD/models/Qwen3-ASR-1.7B"
    export F5TTS_BASE_CKPT="$PWD/models/F5TTS_v1_Base/model_1250000.safetensors"
    export F5TTS_VOCOS_PATH="$PWD/models/vocos-mel-24khz"
    export F5TTS_VOCAB_PATH="$PWD/data/DNS_custom/vocab.txt"

These are repository-relative placeholders. Replace them with paths that exist
on the machine running the experiment.

For WandB logging, authenticate outside the source tree:

    wandb login

or:

    export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

No API key or private server path is required in the repository.

## Training

The project-specific training entry point is:

    src/f5_tts/train/train_reconse.py

The default project configuration is:

    src/f5_tts/configs/ReConSE.yaml

Run training with Accelerate:

    accelerate launch src/f5_tts/train/train_reconse.py --config-name ReConSE

The training script loads a pretrained F5-TTS DiT checkpoint, freezes the
backbone, creates the trainable ControlNet and Qwen adapter modules, and then
starts the dataset and optimizer pipeline.

The configuration controls the dataset name, batch-size policy, optimizer,
mel parameters, ControlNet depth, adapter layers, checkpoint directory, and
local Vocos path. Keep machine-specific overrides in a local untracked
configuration file or provide them through environment variables.

## Inference

The directory inference script accepts a directory of degraded audio and
writes enhanced audio to the requested output directory:

    python src/f5_tts/infer/infer_dir.py \
        --model ReConSE \
        --model_cfg src/f5_tts/configs/ReConSE.yaml \
        --control_audio_dir data/example/noisy \
        --ckpt_file ckpts/model_last.pt \
        --output_dir outputs/enhanced

For a local Vocos checkpoint:

    export F5TTS_VOCOS_PATH="$PWD/models/vocos-mel-24khz"

The upstream F5-TTS vocoder selection remains available where supported. Vocos
is the default for this project-specific workflow.

## Data preparation

Prepared datasets follow the F5-TTS data layout and are configured through
src/f5_tts/configs/ReConSE.yaml. Dataset preparation utilities are available
under src/f5_tts/train/datasets.

Do not commit audio datasets, model checkpoints, generated outputs, WandB
runs, or private evaluation files. The main data, checkpoint, output, and
WandB directories are ignored by git.

## Development checks

Run a basic syntax check before committing:

    python -m compileall src/f5_tts

Check whitespace errors with:

    git diff --check

## Project layout

    src/f5_tts/model/        DiT, CFM, ControlNet, adapters, and datasets
    src/f5_tts/train/        Training entry points and data preparation
    src/f5_tts/infer/        Inference utilities and directory inference
    src/f5_tts/configs/      Hydra model and training configurations
    models/                  Local, untracked model checkpoints
    data/                    Local, untracked datasets
    ckpts/                   Local, untracked training checkpoints
    outputs/                 Local, untracked generated audio

## License and upstream project

This project is based on F5-TTS by SWivid. See LICENSE for the project license
and the upstream repository for the original F5-TTS implementation.
