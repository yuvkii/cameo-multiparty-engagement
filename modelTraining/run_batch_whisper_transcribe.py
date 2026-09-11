"""Transcribe each speaker's separated, aligned audio track (A/B/C/R) per
session via Whisper, for both (1) the text modality itself and (2) as input
to the position<->speaker visual-verification overlay (see
render_speaker_mapping_overlay.py) -- automated audio-visual correlation
(facial-activity, then mouth-landmark motion) was tried first and topped
out too weak to trust (best real-participant correlation ~0.19), so mapping
speaker labels to Cam_2 positions needs a human to confirm it from a
rendered clip instead.

Loads audio via librosa (not ffmpeg -- no system ffmpeg/passwordless sudo
in this environment) and passes the array directly to whisper.transcribe(),
bypassing whisper's normal ffmpeg-based audio loading entirely.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import whisper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "03_20"
SPEAKERS = ["A", "B", "C", "R"]
SESSIONS = [1, 2, 3, 4]


def transcribe_session(model, session: int) -> dict:
    audio_dir = PROJECT_ROOT / "Data" / DATASET / "Processed" / str(session) / "audio" / "aligned_audio"
    result = {}
    for spk in SPEAKERS:
        wav_path = audio_dir / f"{spk}_aligned.wav"
        if not wav_path.exists():
            print(f"  [skip] {wav_path} not found")
            continue
        y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        y = y.astype(np.float32)
        out = model.transcribe(y, language="en")
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in out["segments"]]
        result[spk] = segments
        print(f"  {spk}: {len(segments)} segments")
    return result


if __name__ == "__main__":
    model = whisper.load_model("small")
    for session in SESSIONS:
        print(f"\n=== session {session} ===")
        transcripts = transcribe_session(model, session)
        out_dir = PROJECT_ROOT / "outputs" / DATASET / "transcripts" / f"session_{session}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "speaker_transcripts.json"
        out_path.write_text(json.dumps(transcripts, indent=2))
        print(f"  -> saved {out_path.relative_to(PROJECT_ROOT)}")
