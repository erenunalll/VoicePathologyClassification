import librosa
import numpy as np

def extract_f0(audio_arr: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts fundamental frequency (F0) contour using probabilistic YIN algorithm.
    """

    f0, _, _ = librosa.pyin(audio_arr, 
                            sr = sample_rate, 
                            fmin = librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7')
                            )
    return f0

def extract_local_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate local jitter of an audio signal.
    """

    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0

    if len(periods) < 2:
        return np.nan

    difference = np.abs(np.diff(periods))

    return np.mean(difference) / np.mean(periods)

def extract_rap_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate RAP (Relative Average Perturbation) Jitter of an audio signal.
    """

    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0
    rap_terms = []

    if len(periods) < 3:
        return np.nan

    for i in range(1, len(periods) - 1):
        avg = (periods[i-1] + periods[i] + periods[i+1]) / 3
        rap_terms.append(np.abs(periods[i] - avg))

    return np.mean(rap_terms) / np.mean(periods)

def extract_ppq5_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate PPQ5 (Pitch Perturbation Quotient, 5-Points) Jitter of an audio signal.
    """
    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0
    ppq5_terms = []

    if len(periods) < 5:
        return np.nan
    
    for i in range(2, len(periods) - 2):
        avg = (periods[i-2] + periods[i-1] + periods[i] + periods[i+1] + periods[i+2]) / 5
        ppq5_terms.append(np.abs(periods[i] - avg))
    
    return np.mean(ppq5_terms) / np.mean(periods)