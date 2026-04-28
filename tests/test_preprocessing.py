import numpy as np
import pytest
import pathlib
from src.utils import utils
from src.preprocessing import AudioDataset

def test_trim_on_silence_test():
    audio = AudioDataset(str(pathlib.Path.home)+r"\VoicePathologyClassification\VoicePathologyClassification\data\raw\5 seconds of silence - Ira Bnut (128k).wav",0)
    
    try:
        processed, _ = audio.process()
    except Exception:
        pytest.fail("Process() crashed on silent audio: ", Exception)

def test_output_sample_rate_16kHz():
    audio = AudioDataset(str(pathlib.Path.home)+r"\VoicePathologyClassification\VoicePathologyClassification\data\raw\voice001.wav",0)

    processed, metadata = audio.process()

    assert (metadata["Sampling Rate"] == 16000), (
        "Sampling rate is not 16kHz."
    )

def test_output_amplitude():
    audio = audio = AudioDataset(str(pathlib.Path.home)+r"\VoicePathologyClassification\VoicePathologyClassification\data\raw\voice001.wav",0)
    
    processed, _ = audio.process()
    assert (np.max(processed) <= 1) and (np.min(processed) >= -1), (
        "The audio signal is not normalized."
    )



if __name__ == "__main__" :
    print("-"*50)
    print("UNIT TEST")
    print("-"*50)

    test_trim_on_silence_test()
    test_output_sample_rate_16kHz()
    test_output_amplitude()
    print("-"*50)
