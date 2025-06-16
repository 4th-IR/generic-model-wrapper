from fastapi import UploadFile
from PIL import Image
import numpy as np
import io
import librosa
from transformers import pipeline


def load_image(upload: UploadFile) -> Image.Image:
    return Image.open(upload.file).convert("RGB")


def load_audio(upload: UploadFile, target_sample_rate: int = 16000) -> np.ndarray:
    audio_bytes = upload.file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    waveform, sample_rate = librosa.load(audio_buffer)

    # converting from stereo to mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=target_sample_rate
        )

    return waveform
