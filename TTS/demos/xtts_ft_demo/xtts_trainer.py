import argparse
import os
import sys
import tempfile

import librosa.display
import numpy as np

import os
import torch
import torchaudio
import traceback
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


def train_model(language, train_csv, eval_csv, output_path, num_epochs, batch_size, grad_acumm, max_audio_length):
    clear_gpu_cache()
    if not train_csv or not eval_csv:
        return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
    try:
        # convert seconds to waveform frames
        max_audio_length = int(max_audio_length * 22050)
        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, \
            num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

    # copy original files to avoid parameters changes issues
    os.system(f"cp {config_path} {exp_path}")
    os.system(f"cp {vocab_file} {exp_path}")

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Model training done!")
    clear_gpu_cache()
    return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav


import argparse

def main():
    parser = argparse.ArgumentParser(description="GPU trainer for xtts_v2")
    parser.add_argument("--language", type=str, help="Training for language", default='vi')
    parser.add_argument("--train_csv", type=str, help="Train csv path", default=None)
    parser.add_argument("--eval_csv", type=str, help="Eval csv path", default=None)
    parser.add_argument("--out_path", type=str, help="Output path", default="/tmp/xtts_ft/")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=2)
    parser.add_argument("--grad_acumm", type=int, help="Gradient accumualation", default=8)
    parser.add_argument("--max_audio_length", type=int, help="Max audio length", default=12)
    
    args = parser.parse_args()
    
    train_csv = args.train_csv if args.train_csv else os.path.join(args.out_path, 'dataset', 'metadata_train.csv')
    eval_csv = args.eval_csv if args.eval_csv else os.path.join(args.out_path, 'dataset', 'metadata_eval.csv')

    train_model(args.language, train_csv, eval_csv, args.out_path, \
        args.num_epochs, args.batch_size, args.grad_acumm, args.max_audio_length)
    

if __name__ == "__main__":
    main()
