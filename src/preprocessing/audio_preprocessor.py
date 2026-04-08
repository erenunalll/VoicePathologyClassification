import numpy as np
import librosa
import noisereduce as nr

def normalize(audio_arr):
    peak = np.max(audio_arr)
    return (audio_arr / peak)

def resample16kHz(audio_arr, orig_sampling_rate):
    return librosa.resample(audio_arr, orig_sampling_rate, 16000)

class PreProcessingPipeline:
    def __init__(self, audio_filepath, sampling_rate, prop_decrease):
        self.audio_filepath = audio_filepath
        self.sampling_rate = sampling_rate
        self.prop_decrease = prop_decrease
    
    def process(self):
        voice, sampling_rate = librosa.load(self.audio_filepath, sr=None)
        voice_trimmed, _ = librosa.effects.trim(voice)
        voice_reduced_noise = nr.reduce_noise(voice_trimmed,sr=self.sampling_rate,prop_decrease=self.prop_decrease)
        voice_normalized = normalize(voice_reduced_noise)
        voice_resampled_16kHz = resample16kHz(voice_normalized, self.sampling_rate)
        
        metadata = {"Shape: " : voice_resampled_16kHz.shape,
                    "Sampling Rate: " : "16kHz",
                    "Duration: " : len(voice_resampled_16kHz)/16000,
        }

        return voice_resampled_16kHz, metadata



        


        