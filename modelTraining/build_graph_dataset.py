"""Build per-segment relational graphs for CAMEO's group engagement module.

Each labeled (session, segment) in manual_labeled_dataset.parquet becomes one
small graph (empirically almost always exactly 3 nodes -- 251/281 segments in
03_20; 12 have 2, 18 have 1, none have 4+). Node features are the existing
segment-level scalar aggregates (fk_*/gtp_* from manual_labeled_dataset.parquet).
Edge features are newly constructed here from raw per-frame data -- this is
the actual novel contribution (a real relational signal, not a flat/dummy
adjacency), so it's built as its own step rather than folded into the
existing aggregation scripts.

Two edge sources, chosen per session's camera source (matches
manual_queue.csv's `source` column, same split used throughout this project):

- cam2 sessions (gaze_target, e.g. 3/4 in 03_20): person_idx is NOT a stable
  identity even within one 3-second segment -- confirmed empirically (see
  reidentify_segment docstring). A per-segment re-identification pass runs
  first to get a stable LOCAL id (0/1/2, meaningful only within that one
  segment), then edges are built from real geometry: does person i's
  gaze_peak land inside person j's bbox (direct signal, reuses the same
  containment+margin idea already validated for robot-gaze in
  gaze_target_pipeline.py), plus proximity and mutual-gaze as secondary
  channels.

- rgb sessions (facial_keypoints, e.g. 1/2/5 in 03_20): track_id IS stable
  here, so no re-identification needed. But the pipeline's own who's-
  looking-at-whom column (fk_gaze_target_track_N_frac) is the already-
  flagged-buggy calibrated-gaze signal dropped from the training table
  earlier this project -- not revived here. Instead this uses a coarser,
  honest proxy: does person i's head_yaw_deg sign/magnitude align with the
  on-screen direction toward person j. This is explicitly a weaker signal
  than the cam2 geometric one and should be reported as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).parent.parent

# Same margin idea as gaze_target_pipeline.py's validated robot_gaze_margin_px
# (20px, found necessary because gaze-peak jitter causes systematic near-miss
# containment failures) -- carried over as a reasonable prior for person-bbox
# containment too, though not independently re-validated for this specific case.
GAZE_CONTAINMENT_MARGIN_PX = 20

# Frame-to-frame centroid jump beyond which a detection is treated as a new
# track rather than a continuation of an existing one, during re-identification.
REID_MAX_JUMP_PX = 150

# Minimum |head_yaw_deg| to count a frame as "oriented to one side" at all,
# for the rgb head-pose-alignment edge proxy -- excludes near-neutral
# forward-facing frames from counting as aligned with either neighbor.
HEADPOSE_MIN_YAW_DEG = 8.0


def _centroid(row) -> tuple[float, float]:
    return ((row["bbox_x0"] + row["bbox_x1"]) / 2, (row["bbox_y0"] + row["bbox_y1"]) / 2)


def point_in_bbox(px: float, py: float, box: tuple[float, float, float, float], margin: float = 0) -> bool:
    x0, y0, x1, y1 = box
    return (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin)


def reidentify_segment(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns a stable `local_id` (0, 1, 2, ...) to each row, valid only
    within this one segment's frames -- NOT the same as raw person_idx.

    Confirmed empirically (2026-07-20) that gaze_target's person_idx flips
    identity frequently even within a single 3s/90-frame segment: in
    03_20 session 3 segment 0, every person_idx slot showed a >150px
    frame-to-frame bbox-centroid jump in 24-38 of its 90 frames. Greedy
    frame-to-frame Hungarian matching on centroid distance is a much
    easier problem than session-long tracking and is the right granularity
    anyway, since each segment is its own independent graph instance.
    """
    seg_df = seg_df.sort_values("frame_index").copy()
    local_ids = pd.Series(index=seg_df.index, dtype="Int64")

    active_centroids: dict[int, tuple[float, float]] = {}
    next_id = 0

    for frame_idx, frame_rows in seg_df.groupby("frame_index"):
        det_indices = list(frame_rows.index)
        det_centroids = [_centroid(seg_df.loc[i]) for i in det_indices]

        if not active_centroids:
            for i, c in zip(det_indices, det_centroids):
                local_ids[i] = next_id
                active_centroids[next_id] = c
                next_id += 1
            continue

        track_ids = list(active_centroids.keys())
        track_centroids = [active_centroids[t] for t in track_ids]
        cost = np.zeros((len(det_centroids), len(track_centroids)))
        for a, dc in enumerate(det_centroids):
            for b, tc in enumerate(track_centroids):
                cost[a, b] = np.hypot(dc[0] - tc[0], dc[1] - tc[1])

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= REID_MAX_JUMP_PX:
                tid = track_ids[c]
                local_ids[det_indices[r]] = tid
                active_centroids[tid] = det_centroids[r]
                matched_dets.add(r)

        for a in range(len(det_indices)):
            if a not in matched_dets:
                local_ids[det_indices[a]] = next_id
                active_centroids[next_id] = det_centroids[a]
                next_id += 1

    seg_df["local_id"] = local_ids
    return seg_df


CAM2_NODE_FEATURE_COLS = [
    "n_frames", "center_x_median", "center_y_median",
    "looking_at_robot_rate", "looking_at_participant_rate", "looking_at_elsewhere_rate",
    "gaze_peak_x_std", "gaze_peak_y_std", "gaze_switch_rate",
    "peak_dist_from_self_median", "dist_to_robot",
]

N_OPENFACE_AU = 8
N_OPENFACE_EMOTION = 8
OPENFACE_NODE_FEATURE_COLS = (
    [f"of_au_{i}" for i in range(N_OPENFACE_AU)]
    + [f"of_emotion_{i}" for i in range(N_OPENFACE_EMOTION)]
    + ["of_gaze_yaw", "of_gaze_pitch", "of_availability"]
)


def build_cam2_node_features(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Per-local_id scalar aggregates, recomputed on stable ids (fixes the
    identity-blending that raw person_idx grouping would otherwise cause).

    The original 6 columns (n_frames/center_xy/3 gaze-rate buckets) were
    confirmed too coarse to separate human labels: 45% of near-duplicate
    cam2 node feature vectors (dataset-wide, dist<0.3 after normalization)
    carried conflicting labels, vs 0% for rgb's 124-dim features (2026-07-21).
    Added 5 more columns from data already sitting unused in
    gaze_target's frame_features.csv (gaze_peak_x/y is real per-frame gaze
    point, never aggregated into a node feature before; robot bbox is
    present every frame but was never related to a person's position).
    inout_score was also considered but is 0% populated in this pipeline's
    output (the underlying Gaze-LLE call never returns an inout head here)
    -- not usable without re-running extraction, not silently imputed.
    Empirically cut the near-duplicate/conflict rate to 33% (2328 pairs,
    down from 6717) -- real improvement, not a full fix, since the
    remainder is plausibly genuine human-rater subjectivity rather than
    missing signal."""
    rows = []
    for local_id, g in seg_df.groupby("local_id"):
        cx = (g["bbox_x0"] + g["bbox_x1"]) / 2
        cy = (g["bbox_y0"] + g["bbox_y1"]) / 2
        gx, gy = g["gaze_peak_x"], g["gaze_peak_y"]
        ordered = g.sort_values("frame_index")["gaze_target"].values
        n_switch = sum(1 for a, b in zip(ordered[:-1], ordered[1:]) if a != b)
        robot_cx = (g["robot_x0"] + g["robot_x1"]).mean() / 2
        robot_cy = (g["robot_y0"] + g["robot_y1"]).mean() / 2
        row = {
            "local_id": local_id,
            "n_frames": len(g),
            "center_x_median": cx.median(),
            "center_y_median": cy.median(),
            "gaze_peak_x_std": gx.std() if len(gx) > 1 else 0.0,
            "gaze_peak_y_std": gy.std() if len(gy) > 1 else 0.0,
            "gaze_switch_rate": n_switch / max(len(ordered) - 1, 1),
            "peak_dist_from_self_median": np.hypot(gx - cx, gy - cy).median(),
            "dist_to_robot": np.hypot(cx.median() - robot_cx, cy.median() - robot_cy),
        }
        gaze_counts = g["gaze_target"].value_counts(normalize=True)
        for cat in ("robot", "participant", "elsewhere"):
            row[f"looking_at_{cat}_rate"] = gaze_counts.get(cat, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def merge_openface_features(node_feats: pd.DataFrame, seg_df: pd.DataFrame, openface_df: pd.DataFrame) -> pd.DataFrame:
    """Left-joins Cam_2-native OpenFace 3.0 facial features (AU/emotion/gaze,
    from run_batch_openface_cam2.py) onto the existing gaze/position node
    features, matched via the SAME (frame_index, person_idx) key already
    used by reidentify_segment to assign local_id -- both this segment's
    seg_df and openface_df's rows were matched to gaze_target's own
    person_idx at the same frame, on the same camera, so this join carries
    no cross-camera identity risk (contrast with the abandoned cross-camera
    correlation approach in link_cross_camera_identity.py).

    Per local_id: mean AU activations, mean one-hot emotion distribution
    (softer/more informative than a single dominant-class mode), mean gaze
    angles, and `of_availability` = fraction of this local_id's frames that
    had ANY matched OpenFace row -- many won't (extraction ran at
    frame_step=6, and OpenFace's own detection isn't 100% either), so a
    local_id with zero matched rows gets all-zero OpenFace columns plus
    of_availability=0, letting the model learn to distrust rows with no
    real facial signal rather than silently treating a zero vector as
    "detected neutral face"."""
    if openface_df.empty:
        result = node_feats.copy()
        for col in OPENFACE_NODE_FEATURE_COLS:
            result[col] = 0.0
        return result

    key_map = seg_df[["frame_index", "person_idx", "local_id"]].drop_duplicates()
    joined = openface_df.merge(key_map, on=["frame_index", "person_idx"], how="inner")

    frames_per_local_id = seg_df.groupby("local_id")["frame_index"].nunique()

    agg_rows = []
    for local_id, g in joined.groupby("local_id"):
        row = {"local_id": local_id}
        for i in range(N_OPENFACE_AU):
            row[f"of_au_{i}"] = g[f"au_{i}"].mean()
        emotion_counts = g["emotion_argmax"].value_counts(normalize=True)
        for i in range(N_OPENFACE_EMOTION):
            row[f"of_emotion_{i}"] = emotion_counts.get(i, 0.0)
        row["of_gaze_yaw"] = g["gaze_yaw"].mean()
        row["of_gaze_pitch"] = g["gaze_pitch"].mean()
        n_total = frames_per_local_id.get(local_id, len(g))
        row["of_availability"] = len(g) / max(n_total, 1)
        agg_rows.append(row)
    openface_agg = pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame(columns=["local_id"] + OPENFACE_NODE_FEATURE_COLS)

    result = node_feats.merge(openface_agg, on="local_id", how="left")
    for col in OPENFACE_NODE_FEATURE_COLS:
        result[col] = result[col].fillna(0.0)
    return result


def build_cam2_edges(seg_df: pd.DataFrame, local_ids: list[int],
                      mouth_motion_df: pd.DataFrame | None = None) -> dict[tuple[int, int], dict[str, float]]:
    """Directed edge(i, j) = fraction of i's frames where i's gaze_peak lands
    in j's bbox (+ margin). Also returns proximity and mutual-gaze channels,
    and (2026-07-23) attending_to_speaker: fraction of coexisting frames
    where i's gaze is on j AND j is talking -- a joint-attention/turn-taking
    signal, richer than plain gaze-containment alone, motivated by the graph
    module never showing benefit on the plainer edge channels so far. "Is
    talking" comes from mouth_motion_df (run_batch_mouth_motion.py, visual
    mouth-landmark motion from Cam_2 itself) rather than the separated audio
    channels (A/B/C/R), which were found to have real, session-specific,
    unpredictable crosstalk -- not safe to use as a per-person speech signal.

    "Talking" threshold is the segment's own median mouth_motion across all
    present people/frames, not a fixed global constant -- adapts per-segment
    to that clip's own detection-quality/motion noise floor rather than
    assuming one magnitude works everywhere."""
    by_frame_id: dict[int, dict[int, pd.Series]] = {}
    for frame_idx, g in seg_df.groupby("frame_index"):
        by_frame_id[frame_idx] = {int(row["local_id"]): row for _, row in g.iterrows()}

    talking_by_frame_id: dict[int, dict[int, bool]] = {}
    if mouth_motion_df is not None and not mouth_motion_df.empty:
        key_map = seg_df[["frame_index", "person_idx", "local_id"]].drop_duplicates()
        joined = mouth_motion_df.merge(key_map, on=["frame_index", "person_idx"], how="inner")
        if not joined.empty:
            threshold = joined["mouth_motion"].median()
            for frame_idx, g in joined.groupby("frame_index"):
                talking_by_frame_id[frame_idx] = {int(r.local_id): r.mouth_motion > threshold for r in g.itertuples()}

    edges: dict[tuple[int, int], dict[str, float]] = {}
    for i in local_ids:
        for j in local_ids:
            if i == j:
                continue
            gaze_hits, coexist_frames, dist_sum = 0, 0, 0.0
            attending_hits, attending_frames = 0, 0
            for frame_idx, frame_rows in by_frame_id.items():
                if i not in frame_rows or j not in frame_rows:
                    continue
                coexist_frames += 1
                ri, rj = frame_rows[i], frame_rows[j]
                jbox = (rj["bbox_x0"], rj["bbox_y0"], rj["bbox_x1"], rj["bbox_y1"])
                gaze_hit = point_in_bbox(ri["gaze_peak_x"], ri["gaze_peak_y"], jbox, GAZE_CONTAINMENT_MARGIN_PX)
                if gaze_hit:
                    gaze_hits += 1
                ci, cj = _centroid(ri), _centroid(rj)
                dist_sum += np.hypot(ci[0] - cj[0], ci[1] - cj[1])

                frame_talking = talking_by_frame_id.get(frame_idx)
                if frame_talking is not None and j in frame_talking:
                    attending_frames += 1
                    if gaze_hit and frame_talking[j]:
                        attending_hits += 1

            gaze_rate = gaze_hits / coexist_frames if coexist_frames else 0.0
            mean_dist = dist_sum / coexist_frames if coexist_frames else float("nan")
            attending_rate = attending_hits / attending_frames if attending_frames else 0.0
            edges[(i, j)] = {
                "gaze_rate": gaze_rate, "mean_dist": mean_dist, "coexist_frames": coexist_frames,
                "attending_to_speaker": attending_rate,
            }

    for i in local_ids:
        for j in local_ids:
            if i >= j:
                continue
            gij, gji = edges[(i, j)]["gaze_rate"], edges[(j, i)]["gaze_rate"]
            mutual = min(gij, gji)
            edges[(i, j)]["mutual_gaze"] = mutual
            edges[(j, i)]["mutual_gaze"] = mutual

    return edges


def build_rgb_edges(seg_df: pd.DataFrame, track_ids: list[int]) -> dict[tuple[int, int], dict[str, float]]:
    """Coarser proxy for rgb sessions: does person i's head_yaw sign/magnitude
    align with the on-screen direction toward person j (sign convention
    verified in facial_keypoints_pipeline history: positive yaw = nose
    shifts toward higher screen x). track_id is stable here already."""
    by_frame_track: dict[int, dict[int, pd.Series]] = {}
    for frame_idx, g in seg_df[seg_df["detected"] == 1].groupby("frame_index"):
        by_frame_track[frame_idx] = {int(row["track_id"]): row for _, row in g.iterrows()}

    edges: dict[tuple[int, int], dict[str, float]] = {}
    for i in track_ids:
        for j in track_ids:
            if i == j:
                continue
            aligned, coexist_frames, dist_sum = 0, 0, 0.0
            for frame_rows in by_frame_track.values():
                if i not in frame_rows or j not in frame_rows:
                    continue
                coexist_frames += 1
                ri, rj = frame_rows[i], frame_rows[j]
                ci, cj = _centroid(ri), _centroid(rj)
                dx = cj[0] - ci[0]
                yaw = ri["head_yaw_deg"]
                if abs(yaw) >= HEADPOSE_MIN_YAW_DEG and np.sign(dx) == np.sign(yaw):
                    aligned += 1
                dist_sum += np.hypot(ci[0] - cj[0], ci[1] - cj[1])

            rate = aligned / coexist_frames if coexist_frames else 0.0
            mean_dist = dist_sum / coexist_frames if coexist_frames else float("nan")
            # attending_to_speaker not available for rgb sessions (no mouth-motion
            # extraction built for that camera) -- explicit 0.0, not silently omitted,
            # so the edge tensor's 4th channel stays a defined "no signal" rather than
            # an undefined dict key downstream.
            edges[(i, j)] = {"gaze_rate": rate, "mean_dist": mean_dist, "coexist_frames": coexist_frames,
                              "attending_to_speaker": 0.0}

    for i in track_ids:
        for j in track_ids:
            if i >= j:
                continue
            gij, gji = edges[(i, j)]["gaze_rate"], edges[(j, i)]["gaze_rate"]
            mutual = min(gij, gji)
            edges[(i, j)]["mutual_gaze"] = mutual
            edges[(j, i)]["mutual_gaze"] = mutual

    return edges


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--segment_idx", type=int, required=True)
    args = parser.parse_args()

    queue = pd.read_csv(
        PROJECT_ROOT / "outputs" / args.dataset / "manual_queue.csv", dtype=str, keep_default_na=False
    )
    row = queue[(queue["session"] == str(args.session)) & (queue["segment_idx"] == str(args.segment_idx))].iloc[0]
    source = row["source"]
    start_sec, end_sec = float(row["segment_start_sec"]), float(row["segment_end_sec"])

    if source == "cam2":
        csv_path = PROJECT_ROOT / "outputs" / args.dataset / "gaze_target" / f"session_{args.session}" / "frame_features.csv"
        df = pd.read_csv(csv_path)
        seg = df[(df["timestamp_sec"] >= start_sec) & (df["timestamp_sec"] < end_sec)].dropna(subset=["person_idx"])
        seg = reidentify_segment(seg)
        node_feats = build_cam2_node_features(seg)
        edges = build_cam2_edges(seg, sorted(seg["local_id"].dropna().unique().astype(int)))
        print("nodes:\n", node_feats)
        print("\nedges:")
        for k, v in edges.items():
            print(k, v)
    else:
        csv_path = PROJECT_ROOT / "outputs" / args.dataset / "facial_keypoints" / f"session_{args.session}" / "frame_features.csv"
        df = pd.read_csv(csv_path)
        seg = df[(df["timestamp_sec"] >= start_sec) & (df["timestamp_sec"] < end_sec)]
        track_ids = sorted(seg["track_id"].unique().astype(int))
        edges = build_rgb_edges(seg, track_ids)
        print("edges:")
        for k, v in edges.items():
            print(k, v)
