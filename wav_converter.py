import tensorflow as tf
import numpy as np
import os

def spec_to_wav_tf_array(spec, output_file,
                          frame_length=1024, frame_step=512, fft_length=1024, sample_rate=16000,
                          n_iter=32):
    """
    Convert a magnitude spectrogram to WAV using Griffin-Lim.

    Parameters:
    - spec: 2D numpy array (time x freq), in linear scale or dB
    - output_file: path to save WAV
    - frame_length, frame_step, fft_length: STFT params
    - sample_rate: output WAV sample rate
    - n_iter: number of Griffin-Lim iterations
    """
    # Convert from dB to linear if needed
    if np.max(spec) > 20:  # heuristic: dB > 20
        magnitude = 10 ** (spec / 20.0)
    else:
        magnitude = spec

    magnitude_tf = tf.convert_to_tensor(magnitude, dtype=tf.float32)

    # Griffin-Lim implementation
    def griffin_lim(magnitude, n_iter=n_iter):
        angle = tf.exp(2j * np.pi * tf.random.uniform(tf.shape(magnitude)))
        S = tf.cast(magnitude, tf.complex64) * angle

        for _ in range(n_iter):
            waveform = tf.signal.inverse_stft(
                S,
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=fft_length,
                window_fn=tf.signal.hann_window
            )
            stft = tf.signal.stft(
                waveform,
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=fft_length,
                window_fn=tf.signal.hann_window
            )
            S = tf.cast(magnitude, tf.complex64) * tf.exp(1j * tf.math.angle(stft))

        return waveform

    waveform = griffin_lim(magnitude_tf, n_iter=n_iter)
    waveform = tf.expand_dims(waveform, axis=-1)  # add channel dim

    # Save as WAV
    wav_data = tf.audio.encode_wav(waveform, sample_rate)
    tf.io.write_file(output_file, wav_data)