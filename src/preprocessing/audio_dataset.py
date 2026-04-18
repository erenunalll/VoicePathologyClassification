import numpy as np
import librosa
import noisereduce as nr
from src.utils.utils import *
import os
import pathlib

directoryPath = str(pathlib.Path.home()) + os.sep + "VoicePathologyClassification" + os.sep + "VoicePathologyClassification" + os.sep + "data"


class Preprocess:
    def __init__(self, audio_filename: str, prop_decrease: float):
        self.audio_filename = audio_filename
        self.prop_decrease = prop_decrease
        pass


    def process(self):
        audio_arr, sample_rate = librosa.load(directoryPath + os.sep + "raw" + os.sep + self.audio_filename,sr=None)
        
        
        
        pass

class DataLoader():
    def __init__(self):
        pass

    for root, dirs, files in os.walk(os.path.join(directoryPath, "raw")):
        for file in files:
            pass
                    
        



if __name__ == "__main__":
    x = Preprocess(r"data\raw\148-phrase.wav",0.5)
    voice_resampled, meta= x.process()
    print(meta)


        