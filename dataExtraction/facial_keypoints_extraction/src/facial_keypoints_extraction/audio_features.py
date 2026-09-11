from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(slots=True)
class AudioMetadata:
    path: Path
    sample_rate_hz: int
    channels: int
    duration_sec: float
    sample_width_bytes: int


class StereoAudioFeatureExtractor:
    """Extract lightweight frame-aligned audio descriptors from a WAV file."""

    def __init__(self, wav_path: str | Path) -> None:
        self.wav_path = Path(wav_path)
        with wave.open(str(self.wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            raw = wav_file.readframes(total_frames)

        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are currently supported.")

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, channels)
        self.samples = samples
        self.sample_rate_hz = sample_rate
        self.channels = channels
        self.duration_sec = samples.shape[0] / float(sample_rate)
        self.sample_width_bytes = sample_width

    @property
    def metadata(self) -> AudioMetadata:
        return AudioMetadata(
            path=self.wav_path,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            duration_sec=self.duration_sec,
            sample_width_bytes=self.sample_width_bytes,
        )

    def extract_for_timestamp(
        self,
        timestamp_sec: float,
        window_sec: float = 0.04,
        audio_offset_sec: float = 0.0,
    ) -> dict[str, float]:
        center_index = int((timestamp_sec + audio_offset_sec) * self.sample_rate_hz)
        half_window = max(1, int(window_sec * self.sample_rate_hz / 2.0))
        start = max(0, center_index - half_window)
        stop = min(self.samples.shape[0], center_index + half_window)
        window = self.samples[start:stop]

        if window.size == 0:
            return {
                "audio_rms_left": float("nan"),
                "audio_rms_right": float("nan"),
                "audio_rms_mean": float("nan"),
                "audio_peak_mean": float("nan"),
                "audio_zcr": float("nan"),
                "audio_spectral_centroid_hz": float("nan"),
                "audio_spectral_rolloff_hz": float("nan"),
                "audio_stereo_corr": float("nan"),
                "audio_lr_balance": float("nan"),
            }

        if self.channels == 1:
            left = right = window[:, 0]
        else:
            left = window[:, 0]
            right = window[:, 1]

        mono = window.mean(axis=1)
        rms_left = float(np.sqrt(np.mean(np.square(left))))
        rms_right = float(np.sqrt(np.mean(np.square(right))))
        rms_mean = float(np.sqrt(np.mean(np.square(mono))))
        peak_mean = float(np.max(np.abs(mono)))

        signs = np.signbit(mono)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if mono.size > 1 else 0.0

        spectrum = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / self.sample_rate_hz)
        power = spectrum + 1e-8
        centroid = float(np.sum(freqs * power) / np.sum(power))
        cumulative = np.cumsum(power)
        rolloff_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.85))
        rolloff = float(freqs[min(rolloff_index, freqs.size - 1)])

        if self.channels >= 2 and np.std(left) > 1e-6 and np.std(right) > 1e-6:
            stereo_corr = float(np.corrcoef(left, right)[0, 1])
        else:
            stereo_corr = float("nan")

        balance = float(rms_left - rms_right)

        return {
            "audio_rms_left": rms_left,
            "audio_rms_right": rms_right,
            "audio_rms_mean": rms_mean,
            "audio_peak_mean": peak_mean,
            "audio_zcr": zcr,
            "audio_spectral_centroid_hz": centroid,
            "audio_spectral_rolloff_hz": rolloff,
            "audio_stereo_corr": stereo_corr,
            "audio_lr_balance": balance,
        }
