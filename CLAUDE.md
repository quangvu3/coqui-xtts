# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Coqui XTTS is a fork of Coqui-TTS focused exclusively on XTTS (Cross-lingual Text-to-Speech) functionality. This is a PyTorch-based text-to-speech library that supports voice cloning and multilingual synthesis in 17 languages.

## Installation & Setup
- Python requirements: >= 3.9, < 3.12
- Install with: `pip install -e .[all,dev,notebooks]`
- Dependencies managed via requirements.txt and setup.py
- Uses Cython extensions that need compilation

## Core Architecture

### Directory Structure
- `TTS/` - Main package containing all TTS functionality
  - `tts/` - Text-to-speech models and implementations
    - `models/` - Model definitions (primary: xtts.py)
    - `layers/` - Neural network layer implementations
    - `configs/` - Configuration classes
    - `utils/` - TTS-specific utilities
  - `encoder/` - Speaker encoding functionality
  - `demos/` - Demo applications and examples
    - `xtts_ft_demo/` - Fine-tuning demo with Gradio interface
    - `xtts_oai_server/` - OpenAI-compatible API server
- `utils/` - General utilities for audio processing

### Key Components
- **XTTS Model** (`TTS/tts/models/xtts.py`): Main model class handling synthesis and voice cloning
- **GPT Layer** (`TTS/tts/layers/xtts/gpt.py`): Core language model component
- **HifiDecoder** (`TTS/tts/layers/xtts/hifigan_decoder.py`): Audio generation decoder
- **Tokenizer** (`TTS/tts/layers/xtts/tokenizer.py`): Text preprocessing and tokenization
- **Speaker Manager** (`TTS/tts/layers/xtts/xtts_manager.py`): Multi-speaker voice management

## Development Commands

### Building
- Build package: `python setup.py build_py`
- Install in development mode: `pip install -e .`

### Code Style
- Line length: 120 characters (configured in pyproject.toml)
- Uses Black for formatting with target Python 3.9+
- Uses isort for import sorting
- Flake8 for linting

### Running Demos
- Fine-tuning demo: `python TTS/demos/xtts_ft_demo/xtts_demo.py`
- Training script: `python TTS/demos/xtts_ft_demo/xtts_trainer.py`
- OpenAI server: `python TTS/demos/xtts_oai_server/xtts_server.py`

## Model Usage Patterns

### Voice Cloning Workflow
1. Load model with `Xtts.init_from_config()`
2. Get conditioning latents with `get_conditioning_latents()`
3. Run inference with `inference()` method
4. Save output using torchaudio

### Built-in Speaker Synthesis
1. Load model and checkpoint
2. Use `synthesize()` method with speaker_id parameter
3. Process output tensor for audio saving

## Key Dependencies
- PyTorch >= 2.1 with torchaudio
- transformers >= 4.33.0 for language models
- librosa >= 0.10.0 for audio processing
- soundfile >= 0.12.0 for audio I/O
- trainer >= 0.0.32 for model training
- coqpit >= 0.0.16 for configuration management

## Configuration Management
- Uses Coqpit for type-safe configuration
- XttsConfig class in `TTS/tts/configs/xtts_config.py`
- JSON-based config loading with `load_json()`

## Testing
- No formal test suite detected in current codebase
- Testing appears to be done through demo scripts and manual validation