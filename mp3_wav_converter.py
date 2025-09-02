from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm  # progress bar

# Paths
mp3_path = "clips"
clean_folder = "data/clean"
noisy_folder = "data/noisy"

os.makedirs(clean_folder, exist_ok=True)
os.makedirs(noisy_folder, exist_ok=True)

# Noise level range (min, max)
noise_min = 0.01
noise_max = 0.03

# Function to convert MP3 -> WAV and add random noise
def convert_and_add_noise(mp3_file):
    base_name = os.path.splitext(os.path.basename(mp3_file))[0]

    # Convert to WAV (clean)
    wav_clean_path = os.path.join(clean_folder, f"{base_name}.wav")
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_clean_path, format="wav")

    # Load WAV for noise addition
    y, sr = librosa.load(wav_clean_path, sr=None)

    # Generate a random noise level for this file
    noise_level = np.random.uniform(noise_min, noise_max)
    noise = np.random.normal(0, noise_level, y.shape)
    y_noisy = y + noise

    # Save noisy WAV
    wav_noisy_path = os.path.join(noisy_folder, f"{base_name}_noisy.wav")
    sf.write(wav_noisy_path, y_noisy, sr)

# Process all MP3s with progress bar
files = [f for f in os.listdir(mp3_path) if f.endswith(".mp3")]
print(f"Found {len(files)} MP3 files")

n = int(input("How many files do you want to process? (Enter a number): "))

files = files[:n]

for f in tqdm(files, desc="Converting MP3 → WAV + adding noise"):
    convert_and_add_noise(os.path.join(mp3_path, f))
