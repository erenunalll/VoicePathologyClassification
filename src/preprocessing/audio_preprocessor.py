import numpy as np
import librosa
import noisereduce as nr

class PreProcessingPipeline:
    def __init__(self, audiofile, sample_rate):
        self.audiofile = audiofile
        self.sample_rate = sample_rate
    
    def process(self, filepath):
        audio = librosa.load(filepath,sr=self.sample_rate)
        audio_trim, _ = librosa.effects.trim(audio)
        audio_trim_nr = nr.reduce_noise(audio_trim, prop_decrease=0.5)
        audio_processed = None

        metadata = {"Shape" : None}


        