"""Whole-session person tracking for sessions where the cheap position-rank
identity shortcut (rank 0/1/2 = participant A/B/C, re-derived independently
every frame) is known to be unsafe -- i.e. where people change seats/spots
during the session (e.g. 03_20 session 4, 03_26 session 5), not just sessions
that are position-STABLE (those don't need this at all, plain rank-by-x
already works and is cheaper).

Extends the same core idea already validated in build_graph_dataset.py's
reidentify_segment (greedy Hungarian matching on bbox-centroid distance) from
a single 3s/90-frame segment out to an entire multi-minute session. The
one real addition needed for that jump: reidentify_segment has no velocity
term, which is fine at 3s scale but risks silently swapping two identities
at the exact moment they cross paths over a much longer session (the
"equidistant to both tracks" ambiguity). This version predicts each track's
expected position via a smoothed constant-velocity estimate before matching,
so a person crossing another is matched by CONTINUING THEIR OWN TRAJECTORY,
not by snapping to whoever the nearest static point happens to be.

This is new, unvalidated machinery -- ALWAYS render the debug overlay
(render_tracking_overlay.py) for a session and get it visually confirmed
before trusting local_id output from here for anything that feeds training,
same as this project's established practice for every other perception
pipeline (gaze_target, facial_keypoints, mouth_motion all got a visual
spot-check before being trusted).

Usage (as a library, not run standalone):
    from session_tracker import track_session
    df = pd.read_csv(".../frame_features.csv")
    tracked = track_session(df)  # adds a `local_id` column, stable for the
                                  # whole session (unlike raw person_idx)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Generous walking-pace bound in pixels/second, at this rig's typical
# 2048x1088 framing (~4-5m room width) -- a brisk walk across the room in
# ~2-3s is roughly this order of magnitude. Deliberately generous (favors
# NOT breaking a real track) since this is checked visually before trusting
# it anyway -- tighten only if the overlay shows real identity swaps at
# crossing points, not preemptively.
MAX_SPEED_PX_PER_SEC = 900.0

# How long a track can go completely undetected (occlusion, momentary
# detector miss) before it's considered lost rather than still-trackable.
# Set from real measurement on 03_20 session 4 (2026-07-28): only 3 real
# detection gaps exist in that whole session (one all-3-simultaneous dropout
# at ~3.5s, two later single-person dropouts at ~4s and ~7.9s) -- an initial
# 2.0s value was too short and fragmented one person into 3 separate IDs
# across those gaps. 10s comfortably bridges all of them; if a session has
# a longer real gap this will need revisiting (check track fragmentation
# count against expected ~3 before trusting a new session).
MAX_MISSED_SEC = 10.0

# Beyond this many seconds since a track's last real detection, stop
# extrapolating its predicted position via velocity and freeze at the last
# known position instead. A gap long enough to need MAX_MISSED_SEC's full
# tolerance is also long enough that continuing to project velocity forward
# would likely overshoot -- freezing is the more conservative assumption
# once a person has been unseen for a while.
VELOCITY_FREEZE_AFTER_SEC = 1.0

# Constant-velocity smoothing: new_velocity = ALPHA*raw + (1-ALPHA)*old.
# Lower = smoother/more resistant to single-frame jitter, higher = more
# responsive to genuine direction changes. 0.5 is a straightforward
# middle-ground, not independently tuned.
VELOCITY_SMOOTHING_ALPHA = 0.5

MAX_TRACKS = 3  # this rig always has exactly 3 real participants


def _centroid(row) -> tuple[float, float]:
    return ((row["bbox_x0"] + row["bbox_x1"]) / 2, (row["bbox_y0"] + row["bbox_y1"]) / 2)


class _Track:
    __slots__ = ("track_id", "position", "velocity", "last_frame_idx", "last_ts")

    def __init__(self, track_id: int, position: tuple[float, float], ts: float, frame_idx: int):
        self.track_id = track_id
        self.position = position
        self.velocity = (0.0, 0.0)  # px/sec
        self.last_frame_idx = frame_idx
        self.last_ts = ts

    def predict(self, ts: float) -> tuple[float, float]:
        dt = min(ts - self.last_ts, VELOCITY_FREEZE_AFTER_SEC)
        return (self.position[0] + self.velocity[0] * dt, self.position[1] + self.velocity[1] * dt)

    def update(self, position: tuple[float, float], ts: float, frame_idx: int) -> None:
        dt = ts - self.last_ts
        if dt > 0:
            raw_v = ((position[0] - self.position[0]) / dt, (position[1] - self.position[1]) / dt)
            self.velocity = (
                VELOCITY_SMOOTHING_ALPHA * raw_v[0] + (1 - VELOCITY_SMOOTHING_ALPHA) * self.velocity[0],
                VELOCITY_SMOOTHING_ALPHA * raw_v[1] + (1 - VELOCITY_SMOOTHING_ALPHA) * self.velocity[1],
            )
        self.position = position
        self.last_ts = ts
        self.last_frame_idx = frame_idx


def track_session(df: pd.DataFrame, max_missed_sec: float | None = None) -> pd.DataFrame:
    """max_missed_sec overrides MAX_MISSED_SEC for one session, for a
    genuine long absence rather than a detection wobble -- 05_14 session 3
    has a participant out of frame for 19s (user-confirmed as the same
    person returning, not a swap), which the 10s default correctly refuses
    to bridge, fragmenting them into two tracks. 25s there yields exactly
    3 full-session tracks. Raise it per session on evidence, never
    globally: a longer default would let the tracker paper over real
    identity gaps everywhere else.

    df: one session's raw gaze_target frame_features.csv (every detected
    frame, NOT pre-subsampled -- tracking at the finest available frame rate
    minimizes inter-frame displacement, which is what keeps the matching
    reliable; downstream code can subsample the resulting local_id column
    however it needs to afterward).

    Returns df with a new `local_id` column: 0/1/2 (or higher if more than
    3 simultaneous detections ever occur, e.g. a robot-detection leak into
    the person pool -- see build_graph_dataset.py's own note on this same
    leak). local_id is stable for the WHOLE session, unlike raw person_idx.
    Does NOT map local_id to participant letters A/B/C -- that's a separate,
    explicit step (see map_tracks_to_letters) so the mapping assumption is
    visible and checkable rather than silently baked in here.
    """
    max_missed = MAX_MISSED_SEC if max_missed_sec is None else max_missed_sec
    df = df.dropna(subset=["person_idx", "bbox_x0", "bbox_x1"]).sort_values("frame_index").copy()
    local_ids = pd.Series(index=df.index, dtype="Int64")

    tracks: dict[int, _Track] = {}
    next_id = 0

    for frame_idx, frame_rows in df.groupby("frame_index"):
        ts = float(frame_rows["timestamp_sec"].iloc[0])
        det_indices = list(frame_rows.index)
        det_centroids = [_centroid(df.loc[i]) for i in det_indices]

        # Drop tracks that have been missing too long -- don't let a stale
        # track reach across a big time gap and wrongly re-claim an
        # unrelated later detection.
        for tid in list(tracks.keys()):
            if ts - tracks[tid].last_ts > max_missed:
                del tracks[tid]

        if not tracks:
            for i, c in zip(det_indices, det_centroids):
                if len(tracks) >= MAX_TRACKS:
                    break
                tracks[next_id] = _Track(next_id, c, ts, frame_idx)
                local_ids[i] = next_id
                next_id += 1
            continue

        track_ids = list(tracks.keys())
        predicted = [tracks[t].predict(ts) for t in track_ids]
        cost = np.zeros((len(det_centroids), len(predicted)))
        for a, dc in enumerate(det_centroids):
            for b, pc in enumerate(predicted):
                cost[a, b] = np.hypot(dc[0] - pc[0], dc[1] - pc[1])

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            dt = max(ts - tracks[track_ids[c]].last_ts, 1e-3)
            max_dist = MAX_SPEED_PX_PER_SEC * dt
            if cost[r, c] <= max_dist:
                tid = track_ids[c]
                tracks[tid].update(det_centroids[r], ts, frame_idx)
                local_ids[det_indices[r]] = tid
                matched_dets.add(r)

        for a in range(len(det_indices)):
            if a in matched_dets:
                continue
            if len(tracks) >= MAX_TRACKS:
                continue  # extra simultaneous detection beyond the expected 3 -- likely noise/robot-leak, drop
            tracks[next_id] = _Track(next_id, det_centroids[a], ts, frame_idx)
            local_ids[det_indices[a]] = next_id
            next_id += 1

    df["local_id"] = local_ids
    return df


def map_tracks_to_letters(df: pd.DataFrame, min_timestamp: float = 0.0, seed_window_sec: float = 5.0) -> dict[int, str]:
    """Maps each local_id to A/B/C using majority-vote x-position rank within
    a SHORT WINDOW right after min_timestamp, not the whole session.

    Why a short window, not a single frame or a whole-session vote (history,
    2026-07-28): a single first-frame seed is fragile -- 03_20 session 4's
    very first ~0.4s had a momentarily different left-to-right order than
    the rest of the session (people still taking their seats), which picked
    the wrong track for "A" when trusted literally. Switching to a
    whole-session majority vote fixed that case, but is conceptually WRONG
    in general: this project's continuous_labeler.py explicitly tracks one
    PHYSICAL PERSON for the whole pass, not "whoever's currently on the
    left" -- so the correct letter for a track should be fixed by where
    that physical person started, not by which arrangement happens to hold
    for the most total minutes. This matters concretely: 03_26 session 5 has
    a genuine, sustained ~50/50 seat swap around 109.5s (two people
    physically traded places, confirmed clean via a smoothed rank-crossing
    check -- not noise), so a whole-session majority vote would pick
    whichever half is very slightly longer, silently relabeling ~40-45% of
    the session. A short seed window right after min_timestamp (which itself
    already excludes any pre-clap settling period, per the universal
    clap/sync-point finding -- see continuous_labeling_extension memory)
    gets the TRUE starting arrangement, robust to single-frame noise without
    being swayed by a later swap. Confirmed this also reproduces the correct
    03_20 session 4 mapping (A=stable-left) -- that session has no swap
    after its own min_timestamp, so a short post-cutoff window and a
    whole-session vote agree there; they only diverge when a real swap
    exists, which is exactly the case this needs to get right.
    """
    df = df.dropna(subset=["local_id"])  # extra simultaneous detections beyond MAX_TRACKS are left unassigned
    counts = df.groupby("local_id").size()
    if len(counts) < 3:
        raise ValueError(f"expected 3 tracks, found {len(counts)} -- inspect before mapping")

    window = df[(df["timestamp_sec"] >= min_timestamp) & (df["timestamp_sec"] < min_timestamp + seed_window_sec)]
    if window.empty:
        raise ValueError(f"no frames in the seed window [{min_timestamp}, {min_timestamp + seed_window_sec})")

    rank_votes: dict[int, list[int]] = {}
    for frame_idx, rows in window.groupby("frame_index"):
        if rows["local_id"].nunique() < 3:
            continue
        ranked = rows.assign(cx=(rows["bbox_x0"] + rows["bbox_x1"]) / 2).sort_values("cx")
        for rank, row in enumerate(ranked.itertuples()):
            if rank >= 3:
                break
            rank_votes.setdefault(int(row.local_id), []).append(rank)

    if len(rank_votes) < 3:
        raise ValueError(f"only {len(rank_votes)} tracks co-occurred within the seed window -- "
                          f"widen seed_window_sec or check the window isn't itself unstable")

    mode_rank = {tid: pd.Series(ranks).mode().iloc[0] for tid, ranks in rank_votes.items()}
    letters = ["A", "B", "C"]
    # Break any tie (two tracks with the same mode rank) by mean rank, so the
    # mapping is always fully determined rather than silently dropping one.
    ordered = sorted(mode_rank.items(), key=lambda kv: (kv[1], np.mean(rank_votes[kv[0]])))
    return {tid: letters[i] for i, (tid, _) in enumerate(ordered) if i < 3}
