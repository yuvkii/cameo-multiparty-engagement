"""Build a frame-level (not 3s-segment) graph dataset from continuous
engagement traces (datasetPrep/continuous_labeler.py), for the "test the
continuous scale directly, don't bucket into segments" experiment.

One graph per sampled frame (whoever's detected at that instant, 1-3
nodes), continuous 0-100 label per node taken directly from that
participant's trace AT that exact frame -- no segment averaging. Reuses
the same per-frame data sources and edge-construction logic already
validated in build_graph_dataset.py (gaze containment, proximity, mutual
gaze, attending-to-speaker), just computed instantaneously instead of as
a rate over a 3s window.

Per-person TEMPORAL WINDOW (added 2026-07-30): each node's `node_features`
is a (WINDOW_SIZE, NODE_DIM) sequence -- that participant's own feature
history over the preceding ~3s (see WINDOW_SIZE's docstring) -- not a single
instant. This is what actually uses the temporal granularity the continuous
labeling was built to capture in the first place; previously each frame was
a fully independent example with no memory of a person's own recent past.
Cross-person EDGE features stay computed at the current frame only -- this
matches the design notes' architecture split between "individual
representation" (where a person's history belongs) and "group affective
state" (the current relational structure between people), so the model's
memory lives in the per-node arm, not the graph edges.

Identity reconciliation, two paths depending on the session (see
DATASET_TRACKED_SESSIONS below for which):
- Cheap/default: for sessions confirmed position-STABLE for their whole
  duration (user visual review -- see continuous_labeling_extension /
  continuous_engagement_labeling memory), rank detected person_idx entities
  by on-screen x-position each frame independently; rank 0/1/2 = participant
  A/B/C.
- Tracked: for sessions where people change seats mid-session (position-rank
  alone is unsafe there), a continuous whole-session tracker
  (session_tracker.py) follows each physical person by position+velocity
  instead of re-deriving identity from scratch every frame. Validated
  visually (2026-07-28, 03_20 session 4) before being trusted.
Sessions where NEITHER approach has been validated are excluded entirely
(not guessed at) -- see DATASET_STABLE_RANGES below.

Usage:
    dataExtraction/openface3/.venv/bin/python3 \
        modelTraining/build_continuous_dataset.py --dataset 05_15
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_graph_dataset import GAZE_CONTAINMENT_MARGIN_PX, point_in_bbox
from session_tracker import track_session, map_tracks_to_letters

PROJECT_ROOT = Path(__file__).parent.parent

# Sampling cadence in Hz rather than a fixed frame step: 05_14 is 60fps
# while every other date is 30, so a fixed step would sample it twice as
# densely and silently give it a different temporal resolution from every
# other date in the same pooled model. 5Hz reproduces the old FRAME_STEP=6
# on 30fps footage exactly, so existing dates are unaffected. Must stay in
# step with run_batch_openface_cam2.py / run_batch_mouth_motion.py, which
# use the same constant to decide which frames they extract features for.
TARGET_SAMPLE_HZ = 5.0


def frame_step_for(fps: float) -> int:
    return max(1, int(round(fps / TARGET_SAMPLE_HZ)))


# Per-person temporal window, added 2026-07-30: every comparable engagement-
# estimation system in the literature (MultiMediate/DA-Mamba, DAiSEE-based
# studies) models a person's engagement as evolving over a WINDOW of recent
# frames, not from one instant alone -- this project's own continuous
# labeling was built for exactly that reason (catching within-segment shifts
# a 3s bucket would miss), but the model consuming it was, until now, still
# frame-independent with zero memory of a person's own recent history.
# 15 frames at this ~5fps sampling is ~3s of real time, matching the window
# sizes validated in that literature (e.g. DAiSEE studies commonly use
# 15-16-frame windows). Frames without a FULL window of history available
# (the first ~3s of a stable range) are excluded from being prediction
# targets rather than padded with fabricated history -- a small, honest
# data loss, not an invented signal.
WINDOW_SIZE = 15
POSITIONS = ["A", "B", "C"]
# Lag for the neighbor-engagement edge (added 2026-07-25): a raw correlation
# check (session-by-session, all 5 stable sessions/ranges) showed peer
# engagement correlates most strongly at the SAME instant (mean r=0.64) and
# decays monotonically with lag (0.5s=0.632, 1s=0.612, 2s=0.553, 5s=0.358) --
# consistent with a shared external cause (most likely the robot's own
# behavior, not directly in this model) rather than a peaked lagged-causation
# signature, but still substantial at 1s (0.43-0.68 across every session
# individually, no single outlier driving it). Same-instant can't be used as
# an input feature -- it's the same information the model is being asked to
# predict for the neighbor too. 1s is a reasonable middle ground: strong
# signal, not too degraded, not circular.
LAG_SEC = 1.0
# Deliberately NOT a module constant: it depends on the session's own fps
# (60 for 05_14, 30 elsewhere). Hardcoding 30 would have made 05_14's
# "1 second ago" actually half a second.


def lag_frames_for(fps: float) -> int:
    return int(round(LAG_SEC * fps))

# A 6th edge channel (lagged directed-gaze-rate: did i recently look at j,
# not the same instant) was tried 2026-07-25 and REJECTED -- despite a
# strong standalone raw correlation (mean r=-0.461 outgoing / -0.332
# incoming, negative in 28/30 session-pair combos), adding it to the model
# made every metric slightly WORSE than the 5-edge version below (binary
# acc 81.1% vs 81.2%, 4-level 36.4% vs 37.1%, mae 24.01 vs 23.88, tol10
# 16.8% vs 17.5%). Likely redundant with the existing instantaneous own-
# gaze node feature (sustained gaze behavior over ~2s correlates heavily
# with the current-instant snapshot already in the model) -- see
# cameo_model_build memory for the full writeup. Don't re-add without new
# evidence it helps once that redundancy is addressed some other way.

# (dataset -> {session -> earliest timestamp_sec safe to use}). Position-
# stability is confirmed per-dataset via visual review (see
# continuous_labeling_extension / continuous_engagement_labeling memory) --
# never assume one dataset's table applies to another.
#
# 05_15: session 3 excluded entirely (confirmed unreliable throughout, queue
# scenario). Session 5's opening ~180s excluded (confirmed unreliable, rest
# is fine).
#
# 03_20: sessions 1 and 4 included. Session 2 excluded (real data loss,
# wrong file uploaded, unrelated to position-tracking). Session 3 excluded
# 2026-07-28 (user-directed) after being identified as the main driver of a
# bad combined-training result -- see normalize.py's Normalizer docstring
# for the actual root cause found (cross-date AU-feature-scale mismatch,
# not something specific to session 3's own data quality -- session 3 just
# had the highest true engagement mean, 67.6, making the cross-date bias's
# systematic-underprediction effect most visible there). Session 4 needs the
# TRACKED path, not the cheap rank-based one -- participant A stays reliably
# left all session, but B/C swap places frequently and unpredictably
# throughout, confirmed 2026-07-28 via session_tracker.py + a visually
# reviewed debug overlay (see DATASET_TRACKED_SESSIONS below). Its
# min_timestamp (17.5s) excludes the opening clap/sync-point settling period
# (people not yet in their real positions -- confirmed universal across
# every session in this project, not specific to this one) plus a few
# seconds of tracker reacquisition jitter right after the pre-clap gap.
# Sessions 5/6 don't exist for this date at all.
#
# 03_26: sessions 1/2/3/4/6 confirmed position-stable (2026-07-28). Session 1
# additionally has a real, unrelated data-quality issue: the camera was left
# recording ~47.5s before the session actually started, so its first 1425
# frames (of 5260, ~27%) aren't reliable -- handled as a min_timestamp cutoff
# here rather than editing the recorded continuous_labels CSVs (which are
# gitignored, so a direct edit would be unrecoverable; this is functionally
# identical and matches the 05_15-session-5 precedent for the same kind of
# problem). Session 5 RECOVERED (2026-07-28, see DATASET_TRACKED_SESSIONS
# below) rather than excluded -- initially thought unstable throughout like
# 05_15's session 3, but session_tracker.py + a visual overlay found it's
# actually a clean, sustained ~50/50 seat swap around 109.5s (two people
# genuinely traded places once, not continuous shuffling), fully trackable.
DATASET_STABLE_RANGES = {
    "05_15": {1: 0.0, 2: 0.0, 4: 0.0, 5: 180.0, 6: 0.0},
    # 03_20 session 3: RE-ADDED 2026-08-01. Was swapped out for session 4 on
    # 2026-07-28 (see the 07-28 section of cameo_model_build memory) as part
    # of that day's pooling-diagnosis instructions, not for any data-quality
    # reason -- it was already fully extracted/labeled and confirmed
    # position-stable the whole session (continuous_labeling_extension
    # memory's 03_20 stability table: "3 | yes"), so the rank-based path
    # (no tracker needed) applies, same as session 1. Re-added specifically
    # because 03_20's 2-session count was found to be too thin to support a
    # per-dataset embedding (see model.py's dataset=None backoff comment) --
    # this gives it a 3rd session/2nd sibling per LOSO fold instead of 1.
    "03_20": {1: 0.0, 3: 0.0, 4: 17.5},
    "03_26": {1: 47.5, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    # 05_14 (added 2026-08-03). Cutoffs are DERIVED, not guessed: for each
    # session the tracker was run and the earliest time found from which
    # every track holds a constant x-rank to the end (and, for session 6,
    # the latest time before which it does). That window is exactly the
    # setup/teardown exclusion the user described. Stability per session was
    # independently user-confirmed from labeling all 18 passes: 1/2/4/5/6
    # stable A=left, B=middle, C=right; 3 people move around -> tracked path.
    # The A/B/C-to-position mapping was also verified empirically rather than
    # assumed -- correlating each participant's own trace against each
    # track's looking-at-robot rate gives a clean diagonal on session 2
    # (A->leftmost 0.82, B->middle 0.83, C->rightmost 0.81).
    "05_14": {1: 0.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 17.0, 6: (4.0, 686.0)},
}

# (dataset -> set of sessions) that need session_tracker.py's continuous
# whole-session tracker instead of the cheap per-frame position-rank
# reconciliation -- i.e. sessions in DATASET_STABLE_RANGES above where
# people are known to change seats mid-session. Only add a session here
# after visually confirming the tracker's debug overlay
# (render_tracking_overlay.py) doesn't swap identities -- this was NOT true
# on the first attempt for 03_20 session 4 (an initial 2.0s gap-tolerance
# fragmented one person into multiple track IDs, and a first-frame-only
# letter-mapping picked the wrong track for "A") -- both were real bugs,
# fixed in session_tracker.py, re-verified before trusting.
DATASET_TRACKED_SESSIONS = {
    "03_20": {4},
    "03_26": {5},
    # 05_14 session 3: user-confirmed people move around, so position-rank is
    # unsafe. Needs a 25s gap tolerance (see TRACKER_MAX_MISSED_SEC) because
    # one participant is genuinely out of frame for 19s at 169.9-188.9s --
    # user-confirmed as the same person returning, not a swap. At the 10s
    # default the tracker splits them into two tracks; at 25s the session
    # resolves to exactly 3 tracks each spanning the full duration.
    "05_14": {3},
}

# (dataset -> {session -> seconds}) overriding session_tracker's default gap
# tolerance for one session, on evidence of a real long absence.
TRACKER_MAX_MISSED_SEC = {
    "05_14": {3: 25.0},
}

GAZE_TARGET_CATS = ["robot", "elsewhere", "participant", "unknown"]
N_EMOTION_CLASSES = 8

# NODE_FEATURE_COLS order (29-dim): 4 gaze_target one-hot + center_x/y norm (2)
# + bbox w/h norm (2) + of_availability (1) + face_conf (1) + gaze_yaw/pitch (2)
# + au_0..7 (8) + emotion one-hot (8) + mouth_motion (1) = 29
NODE_DIM = 4 + 2 + 2 + 1 + 1 + 2 + 8 + 8 + 1
EDGE_DIM = 5  # gaze_containment, proximity, mutual_gaze, attending_to_speaker, neighbor_engagement_lagged (added 2026-07-25)


def _load_session_sources(dataset: str, session: int) -> dict:
    base = PROJECT_ROOT / "outputs" / dataset
    gt_path = base / "gaze_target" / f"session_{session}" / "frame_features.csv"
    of_path = base / "openface_cam2" / f"session_{session}" / "frame_features.csv"
    mm_path = base / "mouth_motion" / f"session_{session}" / "frame_features.csv"
    meta_path = base / "gaze_target" / f"session_{session}" / "session_metadata.json"

    gt_df = pd.read_csv(gt_path)
    of_df = pd.read_csv(of_path) if of_path.exists() else pd.DataFrame()
    mm_df = pd.read_csv(mm_path) if mm_path.exists() else pd.DataFrame()

    # Frame resolution varies by dataset (e.g. 03_20/03_26/05_15 are
    # 2048x1088, 05_14 is 1280x1024) -- read it per-session from the gaze_target
    # extraction's own recorded metadata rather than assuming one fixed value
    # (a prior hardcoded 1280x1024 constant was silently wrong for every
    # dataset except 05_14, distorting the position/size node features).
    meta = json.loads(meta_path.read_text())
    frame_w = float(meta["stream"]["frame_width"])
    frame_h = float(meta["stream"]["frame_height"])
    fps = float(meta["stream"]["fps"])

    traces = {}
    for p in POSITIONS:
        trace_path = base / "continuous_labels" / f"session_{session}" / f"participant_{p}.csv"
        traces[p] = pd.read_csv(trace_path).set_index("frame_index")["engagement_pct"] if trace_path.exists() else None

    mm_threshold = mm_df["mouth_motion"].median() if not mm_df.empty else None

    return {"gt": gt_df, "of": of_df, "mm": mm_df, "traces": traces, "mm_threshold": mm_threshold,
            "frame_w": frame_w, "frame_h": frame_h, "fps": fps}


def _node_features(row: pd.Series, of_row: pd.Series | None, mm_value: float | None,
                    frame_w: float, frame_h: float) -> np.ndarray:
    feats = np.zeros(NODE_DIM, dtype=np.float32)
    idx = 0
    for cat in GAZE_TARGET_CATS:
        feats[idx] = 1.0 if row["gaze_target"] == cat else 0.0
        idx += 1
    cx = (row["bbox_x0"] + row["bbox_x1"]) / 2
    cy = (row["bbox_y0"] + row["bbox_y1"]) / 2
    feats[idx] = cx / frame_w; idx += 1
    feats[idx] = cy / frame_h; idx += 1
    feats[idx] = (row["bbox_x1"] - row["bbox_x0"]) / frame_w; idx += 1
    feats[idx] = (row["bbox_y1"] - row["bbox_y0"]) / frame_h; idx += 1

    if of_row is not None:
        feats[idx] = 1.0; idx += 1  # of_availability
        feats[idx] = of_row["face_conf"]; idx += 1
        feats[idx] = of_row["gaze_yaw"]; idx += 1
        feats[idx] = of_row["gaze_pitch"]; idx += 1
        for k in range(8):
            feats[idx] = of_row[f"au_{k}"]; idx += 1
        emotion = int(of_row["emotion_argmax"])
        for k in range(N_EMOTION_CLASSES):
            feats[idx] = 1.0 if k == emotion else 0.0
            idx += 1
    else:
        idx += 1 + 1 + 2 + 8 + N_EMOTION_CLASSES  # leave zero-filled

    feats[idx] = mm_value if mm_value is not None else 0.0; idx += 1
    assert idx == NODE_DIM
    return feats


def _rank_based_letter_lookup(gt_df: pd.DataFrame, min_timestamp: float) -> dict[tuple[int, int], str]:
    """The original, cheap reconciliation: rank detected people by on-screen
    x-position independently every frame, rank 0/1/2 = A/B/C. Only valid for
    sessions confirmed position-STABLE for their whole duration -- see
    DATASET_STABLE_RANGES's docstring-comments for which sessions that is."""
    lookup: dict[tuple[int, int], str] = {}
    sub = gt_df.loc[gt_df["timestamp_sec"] >= min_timestamp].dropna(subset=["person_idx", "bbox_x0", "bbox_x1"])
    for fi, frame_rows in sub.groupby("frame_index"):
        cx = (frame_rows["bbox_x0"] + frame_rows["bbox_x1"]) / 2
        ranked = frame_rows.assign(center_x=cx).sort_values("center_x")
        for rank, row in enumerate(ranked.itertuples()):
            if rank >= len(POSITIONS):
                break
            lookup[(int(fi), int(row.person_idx))] = POSITIONS[rank]
    return lookup


def _tracked_letter_lookup(gt_df: pd.DataFrame, min_timestamp: float,
                            max_missed_sec: float | None = None) -> dict[tuple[int, int], str]:
    """For sessions where people change seats mid-session (position-rank is
    unsafe) but a continuous tracker can still follow each physical person --
    see session_tracker.py. Tracks at the FULL frame rate (not pre-filtered
    to min_timestamp) so the tracker has maximum continuity to work with,
    then the min_timestamp cutoff is applied afterward, same as the
    rank-based path -- min_timestamp is about excluding an unreliable
    stretch from training, not about giving the tracker less to work with."""
    tracked = track_session(gt_df, max_missed_sec=max_missed_sec)
    letters = map_tracks_to_letters(tracked, min_timestamp=min_timestamp)
    valid = tracked.dropna(subset=["local_id"])
    valid = valid[valid["timestamp_sec"] >= min_timestamp]
    return {
        (int(row.frame_index), int(row.person_idx)): letters[int(row.local_id)]
        for row in valid.itertuples()
        if int(row.local_id) in letters
    }


def process_session(dataset: str, session: int, min_timestamp: float, sources: dict,
                     letter_lookup: dict[tuple[int, int], str],
                     max_timestamp: float = float("inf")) -> list[dict]:
    gt_df = sources["gt"]
    of_df, mm_df, traces, mm_threshold = sources["of"], sources["mm"], sources["traces"], sources["mm_threshold"]
    frame_w, frame_h = sources["frame_w"], sources["frame_h"]
    fps = sources["fps"]
    frame_step = frame_step_for(fps)
    lag_frames = lag_frames_for(fps)
    graphs = []

    of_lookup = (of_df.set_index(["frame_index", "person_idx"]) if not of_df.empty else None)
    mm_lookup = (mm_df.set_index(["frame_index", "person_idx"])["mouth_motion"] if not mm_df.empty else None)

    # Real bug found+fixed 2026-07-30: openface_cam2/mouth_motion are extracted
    # per 3s segment (segment boundaries are multiples of 90 frames, itself a
    # multiple of FRAME_STEP=6), so their sampled frame indices always land on
    # a 0-based stride grid (0,6,12,...). Filtering by min_timestamp BEFORE
    # striding (the old `[cond][::FRAME_STEP]`) starts the stride at whatever
    # frame first clears the cutoff, shifting the grid's phase whenever that
    # frame isn't itself a multiple of FRAME_STEP -- e.g. 03_20 session 4's
    # 17.5s cutoff = frame 525 (525 % 6 == 3), so its sampled frames (525,
    # 531, 537, ...) NEVER landed on openface_cam2's (0, 6, 12, ...) grid --
    # confirmed via direct join test: 0 hits out of 200 sampled frames. This
    # silently zero-filled 20 of 29 node-feature dims (of_availability through
    # the last emotion dim) for every node in that session, and the same bug
    # hit 03_26 session 1 (47.5s cutoff = frame 1425, also %6 == 3) -- both
    # measured at 0.000 of_availability before this fix. Fix: stride from the
    # global 0-based grid FIRST, filter by min_timestamp SECOND, so the
    # selected frames always match extraction's grid regardless of cutoff.
    frames = sorted(gt_df["frame_index"].unique())[::frame_step]
    ts_by_frame = gt_df.drop_duplicates("frame_index").set_index("frame_index")["timestamp_sec"]
    frames = [f for f in frames if min_timestamp <= ts_by_frame.loc[f] <= max_timestamp]

    # Pass 1: compute each participant's own per-frame feature vector at
    # every sampled frame they're validly detected at, building a
    # chronologically-sorted history per letter. This is the SAME per-frame
    # computation the old single-frame version did -- just cached per
    # participant instead of being assembled into a graph immediately, since
    # a given frame's features get reused as "history" by every later frame
    # within WINDOW_SIZE steps of it.
    history: dict[str, list[int]] = {p: [] for p in POSITIONS}  # participant -> sorted frame indices
    history_feats: dict[str, dict[int, np.ndarray]] = {p: {} for p in POSITIONS}
    current_frame_rows: dict[int, dict[str, tuple]] = {}  # fi -> {participant: (pid, row)}, for pass 2

    for fi in frames:
        frame_rows = gt_df[(gt_df["frame_index"] == fi)].dropna(subset=["person_idx", "bbox_x0", "bbox_x1"])
        if frame_rows.empty:
            continue
        ranked = frame_rows.reset_index(drop=True)

        for _, row in ranked.iterrows():
            pid = int(row["person_idx"])
            participant = letter_lookup.get((int(fi), pid))
            if participant is None:
                continue
            trace = traces[participant]
            if trace is None or fi not in trace.index:
                continue
            of_row = None
            if of_lookup is not None and (fi, pid) in of_lookup.index:
                of_row = of_lookup.loc[(fi, pid)]
                if isinstance(of_row, pd.DataFrame):  # duplicate match guard
                    of_row = of_row.iloc[0]
            mm_value = None
            if mm_lookup is not None and (fi, pid) in mm_lookup.index:
                mm_value = mm_lookup.loc[(fi, pid)]
                if isinstance(mm_value, pd.Series):
                    mm_value = float(mm_value.iloc[0])

            feat = _node_features(row, of_row, mm_value, frame_w, frame_h)
            history[participant].append(int(fi))
            history_feats[participant][int(fi)] = feat
            current_frame_rows.setdefault(int(fi), {})[participant] = (pid, row)

    def _windowed_feats(participant: str, window_frames: list[int]) -> np.ndarray | None:
        """Hold-last-known-value fill: for each target frame in the window,
        use the most recent history entry at or before it (handles brief
        occlusion/detection dropout within the window without fabricating
        data from nothing). Returns None if the EARLIEST window step has no
        qualifying history at all yet (participant hasn't appeared by then,
        or this is too close to the start of the stable/min_timestamp range)."""
        frame_list = history[participant]
        if not frame_list:
            return None
        idx = bisect.bisect_right(frame_list, window_frames[0]) - 1
        if idx < 0:
            return None  # nothing at/before the earliest window step -- can't fill it, exclude
        seq = np.zeros((WINDOW_SIZE, NODE_DIM), dtype=np.float32)
        for k, t in enumerate(window_frames):
            j = bisect.bisect_right(frame_list, t) - 1
            if j < 0:
                j = idx  # shouldn't happen given the check above, but stay safe
            seq[k] = history_feats[participant][frame_list[j]]
        return seq

    for fi_idx, fi in enumerate(frames):
        if fi not in current_frame_rows or fi_idx < WINDOW_SIZE - 1:
            continue
        window_frames = frames[fi_idx - WINDOW_SIZE + 1: fi_idx + 1]
        ts = float(ts_by_frame.loc[fi])

        node_rows, labels, person_idxs, participants, bboxes = [], [], [], [], []
        for participant, (pid, row) in current_frame_rows[fi].items():
            seq = _windowed_feats(participant, window_frames)
            if seq is None:
                continue  # not enough history yet for this specific person -- exclude just this node
            trace = traces[participant]
            node_rows.append(seq)
            labels.append(float(trace.loc[fi]))
            person_idxs.append(pid)
            participants.append(participant)
            bboxes.append((float(row["bbox_x0"]), float(row["bbox_y0"]), float(row["bbox_x1"]), float(row["bbox_y1"])))

        n = len(node_rows)
        if n == 0:
            continue
        ranked = pd.DataFrame([current_frame_rows[fi][p][1] for p in participants]).reset_index(drop=True)

        edge_tensor = torch.zeros(n, n, EDGE_DIM)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ri = ranked[ranked["person_idx"] == person_idxs[i]].iloc[0]
                rj = ranked[ranked["person_idx"] == person_idxs[j]].iloc[0]
                jbox = (rj["bbox_x0"], rj["bbox_y0"], rj["bbox_x1"], rj["bbox_y1"])
                gaze_hit = point_in_bbox(ri["gaze_peak_x"], ri["gaze_peak_y"], jbox, GAZE_CONTAINMENT_MARGIN_PX)
                ci = ((ri["bbox_x0"] + ri["bbox_x1"]) / 2, (ri["bbox_y0"] + ri["bbox_y1"]) / 2)
                cj = ((rj["bbox_x0"] + rj["bbox_x1"]) / 2, (rj["bbox_y0"] + rj["bbox_y1"]) / 2)
                dist = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                proximity = 1.0 / (1.0 + dist / 500.0)
                edge_tensor[i, j, 0] = 1.0 if gaze_hit else 0.0
                edge_tensor[i, j, 1] = proximity
                j_talking = False
                if mm_lookup is not None and (fi, person_idxs[j]) in mm_lookup.index and mm_threshold is not None:
                    mm_j = mm_lookup.loc[(fi, person_idxs[j])]
                    if isinstance(mm_j, pd.Series):
                        mm_j = float(mm_j.iloc[0])
                    j_talking = mm_j > mm_threshold
                edge_tensor[i, j, 3] = 1.0 if (gaze_hit and j_talking) else 0.0

                # neighbor_engagement_lagged: j's own engagement ~LAG_SEC earlier
                # (0.0 if unavailable, e.g. too close to the start of the
                # session/stable range for a full lag window to exist).
                j_trace = traces[participants[j]]
                lag_fi = fi - lag_frames
                if j_trace is not None and lag_fi in j_trace.index:
                    edge_tensor[i, j, 4] = float(j_trace.loc[lag_fi]) / 100.0
        for i in range(n):
            for j in range(i + 1, n):
                mutual = min(edge_tensor[i, j, 0].item(), edge_tensor[j, i, 0].item())
                edge_tensor[i, j, 2] = mutual
                edge_tensor[j, i, 2] = mutual

        graphs.append({
            "dataset": dataset, "session": session, "frame_index": int(fi), "timestamp_sec": ts,
            "source": "cam2", "label_source": "manual",
            "person_idxs": list(person_idxs), "participants": list(participants), "bboxes": list(bboxes),
            "node_features": torch.tensor(np.stack(node_rows), dtype=torch.float32),
            "edge_features": edge_tensor,
            "labels": torch.tensor(labels, dtype=torch.float32),
        })

    return graphs


def _window(value) -> tuple[float, float]:
    """DATASET_STABLE_RANGES values are either a min timestamp, or a
    (min, max) pair when the END of a session also needs excluding -- 05_14
    sessions are topped and tailed by setup/teardown, where extra people are
    in frame in non-canonical positions (user-flagged 2026-08-03)."""
    if isinstance(value, (tuple, list)):
        return float(value[0]), float(value[1])
    return float(value), float("inf")


def assemble(dataset: str) -> list[dict]:
    if dataset not in DATASET_STABLE_RANGES:
        raise SystemExit(
            f"no position-stability table for dataset {dataset!r} -- add one to "
            f"DATASET_STABLE_RANGES after a visual stability review, don't guess."
        )
    all_graphs = []
    for session, window in DATASET_STABLE_RANGES[dataset].items():
        min_ts, max_ts = _window(window)
        sources = _load_session_sources(dataset, session)
        if session in DATASET_TRACKED_SESSIONS.get(dataset, set()):
            letter_lookup = _tracked_letter_lookup(
                sources["gt"], min_ts,
                max_missed_sec=TRACKER_MAX_MISSED_SEC.get(dataset, {}).get(session),
            )
        else:
            letter_lookup = _rank_based_letter_lookup(sources["gt"], min_ts)
        graphs = process_session(dataset, session, min_ts, sources, letter_lookup, max_timestamp=max_ts)
        print(f"  session {session}: {len(graphs)} graphs")
        all_graphs.extend(graphs)
    return all_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    graphs = assemble(args.dataset)
    out_path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{args.dataset}_continuous.pt"
    torch.save(graphs, out_path)

    node_counts = np.array([g["node_features"].shape[0] for g in graphs])
    all_labels = np.concatenate([g["labels"].numpy() for g in graphs])
    print(f"\n{len(graphs)} graphs total -> {out_path}")
    print(f"node count distribution: {np.bincount(node_counts).tolist()}")
    print(f"label stats: mean={all_labels.mean():.1f} std={all_labels.std():.1f} min={all_labels.min():.1f} max={all_labels.max():.1f}")
