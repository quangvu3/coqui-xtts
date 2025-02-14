
## 🐸 Coqui XTTS
This is a fork of [Coqui-TTS](https://github.com/coqui-ai/TTS) to use XTTS model only.

<div align="center">

## <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>


**🐸 XTTS is a library for advanced Text-to-Speech generation.**

🚀 Pretrained model in 17 languages (*Vietnamese newly added*).

</div>

______________________________________________________________________

## Installation
🐸XTTS is tested on Ubuntu 18.04 up to 24.04 with **python >= 3.9, < 3.12.**.

Clone 🐸XTTS and install it locally.

```bash
git clone https://github.com/quangvu3/coqui-xtts
pip install -e .[all,dev,notebooks]  # Select the relevant extras
```

## Synthesizing speech by 🐸XTTS

### 🐍 Python

#### Running multi-lingual XTTS model

Synthesize speech with a built-in speaker's voice

```python
import os
import torch
import torchaudio
from huggingface_hub import snapshot_download
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

# load configs and model
checkpoint_dir="/path/to/local/checkpoint_dir/"
os.makedirs(checkpoint_dir, exist_ok=True)
repo_id = "jimmyvu/xtts"
snapshot_download(repo_id=repo_id, 
        local_dir=checkpoint_dir, 
        allow_patterns=["*.safetensors", "*.json"])

config = XttsConfig()
config.load_json(os.path.join(checkpoint_dir, "config.json"))
xtts_model = Xtts.init_from_config(config)
xtts_model.load_safetensors_checkpoint(config, checkpoint_dir=checkpoint_dir)

text = "Good morning everyone. I'm an AI model. I can read text and generate speech with a given voice."
language = "en"

# synthesize with speaker id
out = xtts_model.synthesize(text=text, 
			config=xtts_model.config, 
			speaker_wav=None, 
			language=language, 
			speaker_id="Ana Florence")

# save output to wav file
out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
torchaudio.save("speech.wav", out["wav"], 24000)
```

or use a reference speaker (voice cloning)
```python
# reference speaker setup
speaker_audio_file = "/path/to/sample/audio/sample.wav"
gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
	audio_path=speaker_audio_file,
	gpt_cond_len=xtts_model.config.gpt_cond_len,
	max_ref_length=xtts_model.config.max_ref_len,
	sound_norm_refs=xtts_model.config.sound_norm_refs,
)

# inference
out = xtts_model.inference(
	text=text,
	language=language,
	gpt_cond_latent=gpt_cond_latent,
	speaker_embedding=speaker_embedding,
	enable_text_splitting=True,
)

# save output to wav file
out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
torchaudio.save("speech.wav", out["wav"], 24000)
```


## Directory Structure
```
|- utils/           (common utilities.)
|- TTS
    |- tts/             (text to speech models)
        |- layers/          (model layer definitions)
        |- models/          (model definitions)
        |- utils/           (model specific utilities.)
```
