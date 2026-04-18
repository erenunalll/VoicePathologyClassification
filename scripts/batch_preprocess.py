import os
import pathlib 
from scipy.io.wavfile import write
from src.preprocessing import AudioDataset

directoryPath = str(pathlib.Path.home()) + os.sep + "VoicePathologyClassification" + os.sep + "VoicePathologyClassification" + os.sep + "data"
for root, dirs, files in os.walk(os.path.join(directoryPath, "raw")):
    for file in files:
        if not file.lower().endswith(".wav"):
            print("The file " + file + "is not a .wav file and can not be processed.")
            exit(1)
        audio = AudioDataset(os.path.join(root, file),0.5)
        processed_audio, metadata = audio.process()
        
        out_dir = os.path.join(directoryPath,os.path.join("processed", file))
        print(out_dir)
        write(out_dir, 16000, processed_audio)
    print("Directory preprocess complete.")



