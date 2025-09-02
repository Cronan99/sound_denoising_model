# %%
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
import os

import soundfile as sf
import librosa

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tqdm import tqdm

import wav_converter as wc
# %%
# Setting Constants
EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 25
# %%
# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
#%%
# Set memory growth for GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # grow as needed
    except RuntimeError as e:
        print(e)

# %%
# Load Spectrograms
# Paths
clean_spec_folder = "spectrograms/clean"
noisy_spec_folder = "spectrograms/noisy"

# --- Step 1: Find maximum time length across all clean specs ---
def get_target_length(folder, mode="median"):
    lengths = []
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    for f in tqdm(files, desc=f"Measuring lengths in {folder}"):
        spec = np.load(os.path.join(folder, f))
        lengths.append(spec.shape[0])
    
    if mode == "median":
        return int(np.median(lengths))
    elif mode == "p95":  # 95th percentile
        return int(np.percentile(lengths, 95))
    elif mode == "max":
        return max(lengths)
    else:
        raise ValueError("mode must be 'median', 'p95', or 'max'")


max_len = get_target_length(clean_spec_folder, mode="median")
print(f"Padding/truncating all spectrograms to length {max_len}")


# --- Step 2: Collect paired filenames ---
clean_files = sorted([os.path.join(clean_spec_folder, f) for f in os.listdir(clean_spec_folder) if f.endswith(".npy")])
noisy_files = sorted([os.path.join(noisy_spec_folder, f) for f in os.listdir(noisy_spec_folder) if f.endswith(".npy")])
assert len(clean_files) == len(noisy_files), "Mismatch between clean and noisy files"
dataset_size = len(clean_files)
print(f"Dataset size: {dataset_size}")

# --- Step 3: Define a loader function ---
def load_pair(cf, nf):
    clean_spec = np.load(cf.decode())
    noisy_spec = np.load(nf.decode())

    # Pad or truncate
    if clean_spec.shape[0] < max_len:
        pad_width = max_len - clean_spec.shape[0]
        clean_spec = np.pad(clean_spec, ((0, pad_width), (0, 0)), mode='constant')
        noisy_spec = np.pad(noisy_spec, ((0, pad_width), (0, 0)), mode='constant')
    else:
        clean_spec = clean_spec[:max_len, :]
        noisy_spec = noisy_spec[:max_len, :]

    return noisy_spec.astype(np.float32), clean_spec.astype(np.float32)

def tf_load_pair(cf, nf):
    noisy, clean = tf.numpy_function(load_pair, [cf, nf], [tf.float32, tf.float32])
    noisy.set_shape((max_len, 513))
    clean.set_shape((max_len, 513))
    return noisy, clean

# --- Step 4: Build tf.data pipeline ---
batch_size = BATCH_SIZE

dataset = tf.data.Dataset.from_tensor_slices((clean_files, noisy_files))
dataset = dataset.map(tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Split into train/val
train_size = int(0.8 * dataset_size)
train = dataset.take(train_size // batch_size)
val = dataset.skip(train_size // batch_size)

print(f"Dataset ready: {train_size} training samples, {dataset_size - train_size} validation samples")
# %%
# Build the Model
def build_denoising_autoencoder():
    # Input layer - shape: (batch_size, time_steps, frequency_bins)
    inputs = layers.Input(shape=(max_len, 513))
    
    # Add channel dimension for Conv2D
    x = layers.Reshape((max_len, 513, 1))(inputs)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # (max_len/2, 257, 32)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # (max_len/4, 129, 64)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    encoder_output = layers.MaxPooling2D((2, 2), padding='same')(x)  # (max_len/8, 65, 128)
    
    # Get the current shape for TCN
    batch_size, time_steps, freq_bins, channels = encoder_output.shape
    
    # TCN Bottleneck - process time dimension only, keep frequency features as channels
    # Reshape to: (batch_size, time_steps, freq_bins * channels)
    x_flat = layers.Reshape((time_steps, freq_bins * channels))(encoder_output)
    
    # TCN layers - process temporal dimension
    tcn1 = layers.Conv1D(512, 3, padding='same', dilation_rate=1, activation='relu')(x_flat)
    tcn1 = layers.BatchNormalization()(tcn1)
    tcn1 = layers.Dropout(0.1)(tcn1)
    
    tcn2 = layers.Conv1D(512, 3, padding='same', dilation_rate=2, activation='relu')(tcn1)
    tcn2 = layers.BatchNormalization()(tcn2)
    tcn2 = layers.Dropout(0.1)(tcn2)
    
    tcn3 = layers.Conv1D(512, 3, padding='same', dilation_rate=4, activation='relu')(tcn2)
    tcn3 = layers.BatchNormalization()(tcn3)
    tcn3 = layers.Dropout(0.1)(tcn3)
    
    # Project back to original encoder feature dimensions
    tcn_output = layers.Conv1D(freq_bins * channels, 1, padding='same', activation='relu')(tcn3)
    
    # Reshape back to original encoder shape
    x = layers.Reshape((time_steps, freq_bins, channels))(tcn_output)
    
    # Add skip connection from encoder
    x = layers.Add()([x, encoder_output])
    
    # Decoder with careful upsampling to match original dimensions
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Final convolution to get back to original dimensions
    x = layers.Conv2D(1, (3, 3), padding='same', activation='linear')(x)
    
    # Use Cropping2D to ensure exact output dimensions
    # Calculate necessary cropping
    current_height, current_width = x.shape[1], x.shape[2]
    crop_height = current_height - max_len
    crop_width = current_width - 513
    
    # Only crop if needed
    if crop_height > 0 or crop_width > 0:
        crop_top = crop_height // 2
        crop_bottom = crop_height - crop_top
        crop_left = crop_width // 2
        crop_right = crop_width - crop_left
        x = layers.Cropping2D(((crop_top, crop_bottom), (crop_left, crop_right)))(x)
    
    # Remove channel dimension
    outputs = layers.Reshape((max_len, 513))(x)
    
    model = models.Model(inputs, outputs)
    return model

model = build_denoising_autoencoder()
model.summary()
# %%
# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# %%
# Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
best_model = ModelCheckpoint('best_denoising_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    callbacks=[early_stopping, best_model]
)
# %%
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE over epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
# %%
# Save Example Sounds
def spec_to_wav_db(db_spec, output_path, n_fft=1024, hop_length=512, win_length=1024, sr=32000, n_iter=128):
    """
    Convert a dB spectrogram (like the one created by wav_to_spec_tf) back to WAV.
    """
    # Convert dB back to linear magnitude
    linear_spec = 10 ** (db_spec / 20.0)
    
    # Transpose to (freq, time) for librosa
    linear_spec = linear_spec.T
    
    # Reconstruct waveform using Griffin-Lim
    audio = librosa.griffinlim(linear_spec,
                               n_iter=n_iter,
                               hop_length=hop_length,
                               win_length=win_length)
    
    # Save as WAV
    sf.write(output_path, audio, sr)

def save_example_sounds(model, dataset, output_folder="example_sounds", num_samples=5,
                        n_fft=1024, hop_length=512, win_length=1024, sr=22050, n_iter=128):
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (noisy, clean) in enumerate(dataset.take(num_samples)):
        pred = model.predict(noisy)
        
        # Check the shape of the prediction and adjust indexing accordingly
        if pred.ndim == 4:  # Shape: (batch, time, freq, channels)
            pred_spec = pred[0, :, :, 0]  # Take first sample, all time steps, all frequencies, first channel
        elif pred.ndim == 3:  # Shape: (batch, time, freq)
            pred_spec = pred[0, :, :]     # Take first sample, all time steps, all frequencies
        
        noisy_spec = noisy[0, :, :].numpy()
        clean_spec = clean[0, :, :].numpy()

        # Save files
        spec_to_wav_db(noisy_spec, os.path.join(output_folder, f"{i}_noisy.wav"),
                       n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr, n_iter=n_iter)
        spec_to_wav_db(clean_spec, os.path.join(output_folder, f"{i}_clean.wav"),
                       n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr, n_iter=n_iter)
        spec_to_wav_db(pred_spec, os.path.join(output_folder, f"{i}_predicted.wav"),
                       n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr, n_iter=n_iter)

        print(f"Saved sample {i} (noisy, clean, predicted)")

# Example usage
save_example_sounds(
    model=model,
    dataset=val,                  # your validation dataset
    output_folder="example_sounds",
    num_samples=5,                # number of examples to save
    n_fft=1024,
    hop_length=512,
    win_length=1024,
    sr=32000,
    n_iter=128                     # Griffin-Lim iterations for cleaner audio
)

# %%
# Save the model
model.save('denoising_model.h5')
# %%