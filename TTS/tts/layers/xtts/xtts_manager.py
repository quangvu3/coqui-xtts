from pathlib import Path
from safetensors.torch import load_file

class SpeakerManager():
    def __init__(self, speaker_dir=None):
        speaker_files = Path(speaker_dir).rglob('*.safetensors')
        self.speakers = {}
        for f in speaker_files:
            self.speakers[f.stem.replace('_', ' ')] = load_file(f)

    @property
    def name_to_id(self):
        return self.speakers.keys()
    
    @property
    def num_speakers(self):
        return len(self.name_to_id)
    
    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())
    

class LanguageManager():
    def __init__(self, config):
        self.langs = config["languages"]

    @property
    def name_to_id(self):
        return self.langs
    
    @property
    def num_languages(self):
        return len(self.name_to_id)
    
    @property
    def language_names(self):
        return list(self.name_to_id)
