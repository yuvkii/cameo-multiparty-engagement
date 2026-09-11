# facial_keypoints_extraction

`facial_keypoints_extraction` extracts facial landmarks, blendshapes, mouth-opening features, head pose, coarse gaze features, and optional frame-aligned audio descriptors from a ROS1 bag recording.

## Repository layout

Use the repository with this layout:

```text
facial_keypoints_extraction/
|-- data/
|   `-- <session_name>/
|       |-- <session_name>.bag
|       `-- <session_name>.wav
|-- models/
|   `-- face_landmarker.task
|-- outputs/
|   `-- <session_name>/
|-- scripts/
|   |-- extract_multimodal_features.py
|   `-- visualize_landmarks.py
`-- src/
    `-- facial_keypoints_extraction/
```

## Where to put your data

Put each recording session under `data/<session_name>/`.

Recommended naming convention:

- ROS bag: `data/<session_name>/<session_name>.bag`
- Synced audio: `data/<session_name>/<session_name>.wav`
- Output folder: `outputs/<session_name>/`

Example:

```text
data/
`-- session_001/
    |-- session_001.bag
    `-- session_001.wav
```

Notes:

- Keep the `.bag` and `.wav` base name the same when they belong to the same recording.
- The WAV file is optional. If you do not have audio, just omit `--wav`.
- `data/` and `outputs/` are intended for local files and are not synced to GitHub.

## Input requirements

- ROS bag format: ROS1 `.bag`
- Image encoding inside the bag: `bgr8`
- Default topic discovery looks for an RGB topic containing `Color` and ending with `/image/data`
- If automatic topic discovery fails, pass `--color-topic` and `--camera-info-topic` manually

## Environment setup

Create a clean Python environment and install dependencies:

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Model file expected by default:

```text
models/face_landmarker.task
```

## Run feature extraction

Example with both bag and audio:

```powershell
.\.venv312\Scripts\python.exe .\scripts\extract_multimodal_features.py `
  --bag .\data\session_001\session_001.bag `
  --wav .\data\session_001\session_001.wav `
  --output-dir .\outputs\session_001 `
  --frame-step 3 `
  --redetect-interval 5 `
  --save-debug-video
```

Example without audio:

```powershell
.\.venv312\Scripts\python.exe .\scripts\extract_multimodal_features.py `
  --bag .\data\session_001\session_001.bag `
  --output-dir .\outputs\session_001 `
  --frame-step 3 `
  --redetect-interval 5
```

Useful options:

- `--max-frames 120` for a quick smoke test
- `--start-frame 300` to skip the beginning of a session
- `--max-tracks 3` to limit participant slots
- `--audio-offset-sec <value>` if WAV and video timestamps are slightly misaligned
- `--color-topic <topic>` to override RGB image topic discovery
- `--camera-info-topic <topic>` to override camera info topic discovery

## Output files

Running the extractor writes:

- `frame_features.csv`
  - One row per processed frame per participant slot
  - Missing detections remain in the table with `detected=0`
- `landmarks_and_blendshapes.npz`
  - `landmarks`: shape `(T, max_tracks, 478, 3)`
  - `bboxes`: shape `(T, max_tracks, 4)`
  - `blendshapes`: shape `(T, max_tracks, 52)`
- `camera_info.json`
- `session_metadata.json`
- `debug_overlay.mp4` when `--save-debug-video` is enabled

## Validate landmark quality

To render saved landmarks back onto the original frames:

```powershell
.\.venv312\Scripts\python.exe .\scripts\visualize_landmarks.py `
  --bag .\data\session_001\session_001.bag `
  --npz .\outputs\session_001\landmarks_and_blendshapes.npz `
  --output .\outputs\session_001\landmark_validation.mp4 `
  --max-frames 20 `
  --tracks 0 1 2
```

Useful options:

- `--render-stride 2` renders every second NPZ row
- `--tracks 0 1` renders only selected people
- `--mesh-only` draws the face mesh without dense point dots
- `--bbox-only` draws only tracked face boxes and labels

## Current feature set

The current pipeline extracts:

- Face landmarks
- Face blendshapes
- `MAR` mouth-opening feature
- Head pose
- Coarse gaze direction
- Coarse gaze target labels such as `track_0`, `track_1`, `track_2`, `camera`, and `elsewhere`
- Optional lightweight audio frame features from a synced WAV file

The gaze output is intentionally coarse and should be treated as an interaction feature, not a calibrated 3D eye tracker.