import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Paths
clean_wav_folder = "data/clean"
noisy_wav_folder = "data/noisy"
clean_spec_folder = "spectrograms/clean"
noisy_spec_folder = "spectrograms/noisy"

os.makedirs(clean_spec_folder, exist_ok=True)
os.makedirs(noisy_spec_folder, exist_ok=True)

# Parameters
frame_length = 1024
frame_step = 512
fft_length = 1024

def wav_to_spec_tf(wav_file, output_folder):
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    
    # Load WAV with TensorFlow
    audio_binary = tf.io.read_file(wav_file)
    waveform, sr = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)  # remove channels dim
    
    # Compute STFT
    stft = tf.signal.stft(waveform,
                          frame_length=frame_length,
                          frame_step=frame_step,
                          fft_length=fft_length)
    
    # Convert to magnitude in dB 
    magnitude = tf.abs(stft)
    db = 20 * tf.math.log(magnitude + 1e-6) / tf.math.log(10.0)
    
    # Save as NumPy array
    output_file = os.path.join(output_folder, f"{base_name}.npy")
    np.save(output_file, db.numpy())

# Process clean files
clean_files = [f for f in os.listdir(clean_wav_folder) if f.endswith(".wav")]
for f in tqdm(clean_files, desc="Clean spectrograms"):
    wav_to_spec_tf(os.path.join(clean_wav_folder, f), clean_spec_folder)

# Process noisy files
noisy_files = [f for f in os.listdir(noisy_wav_folder) if f.endswith(".wav")]
for f in tqdm(noisy_files, desc="Noisy spectrograms"):
    wav_to_spec_tf(os.path.join(noisy_wav_folder, f), noisy_spec_folder)