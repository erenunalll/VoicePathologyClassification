import numpy as np
import librosa
import noisereduce as nr

def normalize(audio_arr):
    peak = np.max(audio_arr)
    return (audio_arr / peak)

def resample16kHz(audio_arr, orig_sampling_rate):
    return librosa.resample(audio_arr, orig_sr=orig_sampling_rate, target_sr=16000)

class AudioDataset:


    def __init__(self, audio_filepath, prop_decrease):
        self.data = []
        self.metadata = []
        self.audio_filepath = audio_filepath
        # self.prop_decrease = prop_decrease
        
        #TODO : Read all vaw files in given audio_filepath
        #Add it to data and metadata 
        #self.process(i)

        #TODO GET function to get a data at index i and metadata i 







    def process(self,audio_sample):
        voice, sampling_rate = librosa.load(self.audio_filepath, sr=None)
        voice_trimmed, _ = librosa.effects.trim(voice)
        
        duration = len(voice_trimmed)/sampling_rate

        voice_reduced_noise = nr.reduce_noise(voice_trimmed,sr=sampling_rate,prop_decrease=self.prop_decrease)
        voice_normalized = normalize(voice_reduced_noise)
        voice_resampled_16kHz = resample16kHz(voice_normalized, sampling_rate)
        
        resampled_sr = 16000
        duration_resampled = len(voice_resampled_16kHz) / resampled_sr
        
        metadata = {"Shape" : voice_resampled_16kHz.shape,
                    "Sampling Rate" : resampled_sr,
                    "Duration" : duration_resampled,
        }

        return voice_resampled_16kHz, metadata

    def __str__(self):
        pass

if __name__ == "__main__":
    x = AudioDataset(r"data\raw\148-phrase.wav",0.5)
    voice_resampled, meta= x.process()
    print(meta)


        