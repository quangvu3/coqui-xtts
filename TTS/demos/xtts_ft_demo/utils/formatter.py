import os
import gc
import torchaudio
import pandas
from faster_whisper import WhisperModel
from glob import glob

from tqdm import tqdm

import torch
import torchaudio
# torch.set_num_threads(1)

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners, split_sentence

torch.set_num_threads(16)


import os

audio_types = (".wav", ".mp3", ".flac")


def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath


def split_text_by_rules(text, max_chars=250):
    """
    Split text following breaking rules:
    1. Punctuation marks (. ! ?)
    2. Commas (,)
    3. Character limit (250 chars, whole words)
    
    Args:
        text (str): Text to split
        max_chars (int): Maximum characters per chunk
    
    Returns:
        list: List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences first (punctuation)
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Process sentences
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Sentence is too long, split by commas
                comma_parts = sentence.split(',')
                for part in comma_parts:
                    part = part.strip()
                    if len(current_chunk) + len(part) <= max_chars:
                        current_chunk += (", " + part if current_chunk else part)
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = part
                        else:
                            # Part is still too long, split by words
                            words = part.split()
                            for word in words:
                                if len(current_chunk) + len(word) + 1 <= max_chars:
                                    current_chunk += (" " + word if current_chunk else word)
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                        current_chunk = word
                                    else:
                                        # Single word is too long
                                        chunks.append(word)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_text_evenly(text, num_chunks):
    """
    Split text into approximately equal chunks.
    
    Args:
        text (str): Text to split
        num_chunks (int): Number of chunks to create
    
    Returns:
        list: List of text chunks
    """
    if num_chunks <= 1:
        return [text]
    
    words = text.split()
    words_per_chunk = len(words) // num_chunks
    remainder = len(words) % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        chunk_size = words_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]
        chunks.append(" ".join(chunk_words))
        start_idx = end_idx
    
    return chunks

def find_chunk_split_point(words_list, start_idx, max_duration, max_chars=250):
    """
    Find the best split point to create a chunk that fits within 11 seconds,
    using breaking rules in priority order:
    1. Punctuation marks (. ! ?)
    2. Commas (,)  
    3. 250 characters (whole words)
    4. Duration limit (emergency)
    
    Args:
        words_list (list): List of word objects with timestamps
        start_idx (int): Starting word index
        max_duration (float): Maximum duration in seconds (11s)
        max_chars (int): Maximum characters before forcing split (250)
    
    Returns:
        int: Word index where to end the chunk, or -1 if no words available
    """
    if start_idx >= len(words_list):
        return -1
    
    start_time = words_list[start_idx].start
    max_end_time = start_time + max_duration
    
    current_text = ""
    last_punctuation_idx = -1
    last_comma_idx = -1
    last_valid_idx = -1
    
    # Go through words and find potential split points
    for i in range(start_idx, len(words_list)):
        word = words_list[i]
        
        # Check if this word would exceed time limit
        if word.end > max_end_time:
            break
            
        current_text += word.word
        last_valid_idx = i
        
        # Rule 1: Track punctuation marks (. ! ?)
        if word.word.strip().endswith(('.', '!', '?')):
            last_punctuation_idx = i
        
        # Rule 2: Track commas
        elif word.word.strip().endswith(','):
            last_comma_idx = i
        
        # Rule 3: Check 250 character limit
        if len(current_text) >= max_chars:
            # Prefer punctuation > comma > current position
            if last_punctuation_idx >= start_idx:
                return last_punctuation_idx
            elif last_comma_idx >= start_idx:
                return last_comma_idx
            else:
                return i  # Split at current word to stay under character limit
    
    # If we processed all words without hitting limits, use best available split
    if last_punctuation_idx >= start_idx:
        return last_punctuation_idx
    elif last_comma_idx >= start_idx:
        return last_comma_idx
    elif last_valid_idx >= start_idx:
        return last_valid_idx
    
    return -1


def format_audio_list(audio_files, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", max_duration=11.0, gradio_progress=None):
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    print("Loading Whisper Model...")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, 
                                           word_timestamps=False, 
                                           language=target_language,
                                           condition_on_previous_text=False,  # Reduce repetition
                                           log_prob_threshold=-1.0,           # Filter low-probability tokens
                                           no_speech_threshold=0.6            # Better silence detection
        )
        segments = list(segments)
        i = 0  # Initialize chunk counter
        filtered_count = 0  # Track filtered segments
        
        # Process segments by creating chunks that fit within 11 seconds
        # Since we don't have word timestamps, we'll work with segment-level data
        for segment_idx, segment in enumerate(segments):
            segment_text = segment.text.strip()
            if not segment_text:
                continue
            
            # Filter out unwanted promotional content
            promotional_phrases = [
                'subscribe cho kênh ghiền mì gõ',
                'ghiền mì gõ',
                'đăng ký kênh',
                'like và subscribe'
            ]
            
            should_skip = False
            for phrase in promotional_phrases:
                if phrase in segment_text.lower():
                    print(f"Skipping promotional content: {segment_text[:50]}...")
                    should_skip = True
                    break
            
            if should_skip:
                filtered_count += 1
                continue
            
            segment_start = segment.start
            segment_end = segment.end
            segment_duration = segment_end - segment_start
            
            # If segment is already short enough, use it as-is
            if segment_duration <= max_duration:
                # Apply breaking rules to potentially split long text within the segment
                chunks = split_text_by_rules(segment_text, max_chars=250)
                
                if len(chunks) == 1:
                    # Single chunk - use whole segment
                    chunk_text = multilingual_cleaners(segment_text, target_language)
                    
                    # Calculate timing with buffer
                    chunk_start_time = segment_start
                    chunk_end_time = segment_end
                    
                    if segment_idx == 0:
                        chunk_start_time = max(chunk_start_time - buffer, 0)
                    else:
                        prev_end = segments[segment_idx - 1].end
                        chunk_start_time = max(chunk_start_time - buffer, (prev_end + chunk_start_time) / 2)
                    
                    if segment_idx == len(segments) - 1:
                        chunk_end_time = min(chunk_end_time + buffer, (wav.shape[0] - 1) / sr)
                    else:
                        next_start = segments[segment_idx + 1].start
                        chunk_end_time = min((chunk_end_time + next_start) / 2, chunk_end_time + buffer)
                    
                    # Save chunk
                    if chunk_text and (chunk_end_time - chunk_start_time) >= 1/3:  # Min duration check
                        audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                        audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"
                        absolute_path = os.path.join(out_path, audio_file)
                        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                        
                        audio = wav[int(sr * chunk_start_time):int(sr * chunk_end_time)].unsqueeze(0)
                        
                        if audio.size(-1) >= sr / 3:
                            duration = chunk_end_time - chunk_start_time
                            print(f"Chunk {i}: {duration:.2f}s, {len(chunk_text)} chars, text: {chunk_text[:50]}...")
                            
                            torchaudio.save(absolute_path, audio, sr)
                            metadata["audio_file"].append(audio_file)
                            metadata["text"].append(chunk_text)
                            metadata["speaker_name"].append(speaker_name)
                            i += 1
                
                else:
                    # Multiple text chunks - split segment proportionally by time
                    total_chars = len(segment_text)
                    cumulative_chars = 0
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_text = multilingual_cleaners(chunk_text, target_language)
                        chunk_chars = len(chunk_text)
                        
                        # Calculate proportional timing
                        start_ratio = cumulative_chars / total_chars
                        end_ratio = (cumulative_chars + chunk_chars) / total_chars
                        
                        chunk_start_time = segment_start + (segment_duration * start_ratio)
                        chunk_end_time = segment_start + (segment_duration * end_ratio)
                        
                        # Add buffers
                        if segment_idx == 0 and chunk_idx == 0:
                            chunk_start_time = max(chunk_start_time - buffer, 0)
                        if segment_idx == len(segments) - 1 and chunk_idx == len(chunks) - 1:
                            chunk_end_time = min(chunk_end_time + buffer, (wav.shape[0] - 1) / sr)
                        
                        # Save chunk
                        if chunk_text and (chunk_end_time - chunk_start_time) >= 1/3:
                            audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                            audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"
                            absolute_path = os.path.join(out_path, audio_file)
                            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                            
                            audio = wav[int(sr * chunk_start_time):int(sr * chunk_end_time)].unsqueeze(0)
                            
                            if audio.size(-1) >= sr / 3:
                                duration = chunk_end_time - chunk_start_time
                                print(f"Chunk {i}: {duration:.2f}s, {len(chunk_text)} chars, text: {chunk_text[:50]}...")
                                
                                torchaudio.save(absolute_path, audio, sr)
                                metadata["audio_file"].append(audio_file)
                                metadata["text"].append(chunk_text)
                                metadata["speaker_name"].append(speaker_name)
                                i += 1
                        
                        cumulative_chars += chunk_chars
            
            else:
                # Segment is too long - force split into multiple chunks of max_duration
                chunks_needed = int(segment_duration / max_duration) + 1
                chunk_duration = segment_duration / chunks_needed
                
                # Split text proportionally
                text_chunks = split_text_by_rules(segment_text, max_chars=250)
                if len(text_chunks) < chunks_needed:
                    # Force more splits if needed
                    text_chunks = split_text_evenly(segment_text, chunks_needed)
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_text = multilingual_cleaners(chunk_text, target_language)
                    
                    # Calculate timing
                    chunk_start_time = segment_start + (chunk_idx * chunk_duration)
                    chunk_end_time = min(segment_start + ((chunk_idx + 1) * chunk_duration), segment_end)
                    
                    # Add buffers
                    if segment_idx == 0 and chunk_idx == 0:
                        chunk_start_time = max(chunk_start_time - buffer, 0)
                    if segment_idx == len(segments) - 1 and chunk_idx == len(text_chunks) - 1:
                        chunk_end_time = min(chunk_end_time + buffer, (wav.shape[0] - 1) / sr)
                    
                    # Save chunk
                    if chunk_text and (chunk_end_time - chunk_start_time) >= 1/3:
                        audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                        audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"
                        absolute_path = os.path.join(out_path, audio_file)
                        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                        
                        audio = wav[int(sr * chunk_start_time):int(sr * chunk_end_time)].unsqueeze(0)
                        
                        if audio.size(-1) >= sr / 3:
                            duration = chunk_end_time - chunk_start_time
                            print(f"Chunk {i}: {duration:.2f}s, {len(chunk_text)} chars, text: {chunk_text[:50]}...")
                            
                            torchaudio.save(absolute_path, audio, sr)
                            metadata["audio_file"].append(audio_file)
                            metadata["text"].append(chunk_text)
                            metadata["speaker_name"].append(speaker_name)
                            i += 1

    # Post-processing validation and statistics
    print(f"\n=== Audio Segmentation Results ===")
    print(f"Total segments created: {len(metadata['audio_file'])}")
    print(f"Promotional segments filtered: {filtered_count}")
    print(f"All segments processed successfully!\n")

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size