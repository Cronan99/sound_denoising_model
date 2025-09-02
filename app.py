import os
import io
import zipfile
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from keras.layers import TFSMLayer, Input
from keras.models import Model
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# ==========================
# Parameters (match training except time-steps, which we'll detect)
# ==========================
FRAME_LENGTH = 1024
FRAME_STEP   = 512
FFT_LENGTH   = 1024
SR           = 32000
FREQ_BINS    = 513
N_ITER       = 128

# ==========================
# Model download / load (SavedModel via TFSMLayer)
# ==========================
MODEL_DIR     = "best_denoising_savedmodel"
MODEL_ZIP_URL = "https://github.com/Cronan99/sound_denoising_model/releases/download/model/best_denoising_savedmodel.zip"

def download_and_unzip(url: str, extract_into: str = "."):
    os.makedirs(extract_into, exist_ok=True)
    with st.spinner("Downloading model..."):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(extract_into)

def _detect_required_timesteps(savedmodel_dir: str, endpoint: str = "serving_default") -> int:
    """Loads the SavedModel signature and returns required time length (shape[1])."""
    sm = tf.saved_model.load(savedmodel_dir)
    sig = sm.signatures.get(endpoint)
    if sig is None:
        raise ValueError(f"No signature '{endpoint}' found in SavedModel.")
    # structured_input_signature = (args_spec, kwargs_spec); we need kwargs dict
    kwargs = sig.structured_input_signature[1]
    # Take the first (and usually only) input spec
    inp_spec = next(iter(kwargs.values()))
    # shape is usually (None, time, freq)
    time_steps = int(inp_spec.shape[1])
    freq_bins  = int(inp_spec.shape[2])
    if freq_bins != FREQ_BINS:
        raise ValueError(f"SavedModel expects {freq_bins} freq bins, but app is set to {FREQ_BINS}.")
    return time_steps

@st.cache_resource
def load_model_and_shape():
    # Ensure SavedModel folder exists locally
    if not os.path.isdir(MODEL_DIR):
        download_and_unzip(MODEL_ZIP_URL, ".")

    # Detect required time length from signature (e.g., 390)
    required_len = _detect_required_timesteps(MODEL_DIR, endpoint="serving_default")

    # Wrap SavedModel for Keras 3
    tfsmlayer = TFSMLayer(MODEL_DIR, call_endpoint="serving_default")

    # Build a Keras wrapper so we can call model.predict(...)
    x_in  = Input(shape=(required_len, FREQ_BINS), dtype=tf.float32, name="spec_input")
    y_out = tfsmlayer(x_in)  # may return a tensor or a dict
    if isinstance(y_out, dict):   # if SavedModel returns a dict, take first value
        y_out = list(y_out.values())[0]
    wrapped = Model(inputs=x_in, outputs=y_out, name="denoiser_wrapper")
    return wrapped, required_len

model, MAX_LEN = load_model_and_shape()   # <- MAX_LEN comes from the SavedModel (e.g., 390)

# ==========================
# Utility functions
# ==========================
def wav_to_spec_db(y, sr=SR):
    stft = librosa.stft(y, n_fft=FFT_LENGTH, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
    magnitude = np.abs(stft)
    db = 20 * np.log10(magnitude + 1e-6)
    return db.T  # (time, freq)

def pad_or_truncate(spec, max_len=None):
    if max_len is None:
        max_len = MAX_LEN
    if spec.shape[0] < max_len:
        pad_width = max_len - spec.shape[0]
        spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant')
    else:
        spec = spec[:max_len, :]
    return spec

def spec_to_wav_db(db_spec, sr=SR):
    linear_spec = 10 ** (db_spec / 20.0)
    linear_spec = linear_spec.T
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=N_ITER,
        hop_length=FRAME_STEP,
        win_length=FRAME_LENGTH
    )
    return audio

def add_noise(y, noise_min=0.01, noise_max=0.03):
    noise_level = np.random.uniform(noise_min, noise_max)
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def plot_spectrogram(spec, title="Spectrogram"):
    fig, ax = plt.subplots()
    im = ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Frequency bins")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

# ==========================
# Streamlit UI
# ==========================
st.title("Speech Denoising Demo")

uploaded_file = st.file_uploader("Upload a CLEAN audio file (wav, mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load CLEAN audio at SR=32000
    y_clean, sr = librosa.load(uploaded_file, sr=SR)
    st.subheader("Uploaded clean audio")
    st.audio(uploaded_file, format="audio/wav")

    # Create synthetic noisy version (unchanged path)
    y_noisy = add_noise(y_clean)

    # Convert to spectrograms
    spec_clean = wav_to_spec_db(y_clean, sr=SR)          # for visualization
    spec_noisy = wav_to_spec_db(y_noisy, sr=SR)
    spec_proc  = pad_or_truncate(spec_noisy, MAX_LEN)    # <- uses detected required length

    # Predict
    input_tensor = np.expand_dims(spec_proc, axis=0).astype(np.float32)  # (1, time, freq)
    pred_spec    = model.predict(input_tensor)[0]

    # Convert back to audio (UNCHANGED: reconstruct both noisy & predicted)
    audio_noisy = spec_to_wav_db(spec_proc, sr=SR)
    audio_pred  = spec_to_wav_db(pred_spec, sr=SR)

    # Save temp files
    noisy_path = "noisy_out.wav"
    pred_path  = "denoised_out.wav"
    sf.write(noisy_path, audio_noisy, SR)
    sf.write(pred_path,  audio_pred,  SR)

    # Play back
    st.subheader("Results")
    st.write(f"Noisy input (reconstructed from padded spectrogram; length={MAX_LEN} frames):")
    st.audio(noisy_path, format="audio/wav")
    st.write("Denoised output (model prediction):")
    st.audio(pred_path, format="audio/wav")

    # Spectrograms
    st.subheader("Spectrograms")
    plot_spectrogram(spec_clean, title="Clean Spectrogram (dB)")
    plot_spectrogram(spec_noisy, title="Noisy Spectrogram (dB)")
    plot_spectrogram(pred_spec, title="Predicted (Denoised) Spectrogram (dB)")
