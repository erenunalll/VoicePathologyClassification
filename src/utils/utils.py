import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt

def silence_trim(audio_arr: np.ndarray, top_db: int):
    if (audio_arr[audio_arr < top_db].sum() == len(audio_arr)):
        return audio_arr

    return librosa.effects.trim(audio_arr,top_db=top_db)[0]

def reduce_noise(audio_arr: np.ndarray, sample_rate: int, prop_decrease: float):
    return nr.reduce_noise(audio_arr,sr=sample_rate,prop_decrease=prop_decrease)

def peak_normalization(audio_arr: np.ndarray):
    return (audio_arr / np.max(audio_arr))

def resample(audio_arr: np.ndarray, sample_rate: int, target_sr: int):
    return librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=target_sr)

def waveform(audio_arr: np.ndarray, sample_rate: int):
    t = np.linspace(0, len(audio_arr)/sample_rate,num=len(audio_arr))
    
    plt.figure(figsize=(15,4))
    plt.plot(t, audio_arr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def spectogram(audio_arr: np.ndarray,sample_rate: int):
    audio_stft = librosa.stft(audio_arr)
    audio_db = librosa.amplitude_to_db(audio_stft)
    
    plt.figure(figsize=(15,4))
    librosa.display.specshow(audio_db,sr=sample_rate,x_axis="time",y_axis="hz",cmap="magma")
    plt.title("Spectogram")
    plt.colorbar(format="%2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    