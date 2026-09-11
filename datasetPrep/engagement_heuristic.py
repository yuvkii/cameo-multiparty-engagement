"""Heuristic engagement auto-labeling from already-extracted segment features.

No new model/API dependency -- scores each segments.parquet row from
existing columns, using "how much gaze was directed at the robot" as the
core signal, available from two independent sources per segment:

- gt_looking_at_robot_rate (mocap gaze_target pipeline) -- the more
  trustworthy source; two real bugs in this pipeline were found and fixed
  (participant classification, robot-bbox margin), both visually validated.
- fk_gaze_target_camera_frac (facial_keypoints pipeline's own calibrated-
  angle gaze column, filtered to the "camera"=~robot category) -- has a
  known unfixed bug (attach_gaze_targets: participant-match can wrongly
  win over a robot-match), so weighted lower.

Row granularity follows whatever segments.parquet already has: per
(session, segment, track_id) when facial_keypoints covers that session
(individual-level), per (session, segment) group-level otherwise.
"""
from __future__ import annotations

import pandas as pd

_GT_WEIGHT = 0.7
_FK_WEIGHT = 0.3

ENGAGEMENT_LEVELS = ("disengaged", "low", "medium", "high")

REVIEW_FRACTION = 0.20


def compute_engagement_proxy(df: pd.DataFrame) -> pd.Series:
    gt = df.get("gt_looking_at_robot_rate")
    fk = df.get("fk_gaze_target_camera_frac")

    if gt is None and fk is None:
        raise ValueError("Neither gt_looking_at_robot_rate nor fk_gaze_target_camera_frac present")

    # Percentile-rank each signal within its own population before blending.
    # Raw values are NOT comparable: gt_looking_at_robot_rate typically runs
    # 0.57-0.78, fk_gaze_target_camera_frac 0.07-0.24 (verified on 03_20) --
    # fk's stricter calibrated-angle "camera" category is inherently rarer
    # than gt's looser bbox-containment check. Blending raw fractions would
    # systematically drag any fk-only-covered segment toward "disengaged"
    # purely from scale, not real behavior.
    gt_rank = gt.rank(pct=True) if gt is not None else None
    fk_rank = fk.rank(pct=True) if fk is not None else None

    if gt_rank is None:
        return fk_rank.copy()
    if fk_rank is None:
        return gt_rank.copy()

    both = gt_rank.notna() & fk_rank.notna()
    proxy = pd.Series(index=df.index, dtype=float)
    proxy[both] = _GT_WEIGHT * gt_rank[both] + _FK_WEIGHT * fk_rank[both]
    only_gt = gt_rank.notna() & ~fk_rank.notna()
    proxy[only_gt] = gt_rank[only_gt]
    only_fk = fk_rank.notna() & ~gt_rank.notna()
    proxy[only_fk] = fk_rank[only_fk]
    return proxy


def assign_engagement_level(proxy_scores: pd.Series) -> tuple[pd.Series, list[float]]:
    valid = proxy_scores.dropna()
    if valid.empty:
        raise ValueError("No valid proxy scores to compute thresholds from")

    # Data-driven quartile boundaries -- 3 cut points split into 4 levels.
    thresholds = valid.quantile([0.25, 0.5, 0.75]).tolist()

    def _level(score: float) -> object:
        if pd.isna(score):
            return pd.NA
        if score <= thresholds[0]:
            return ENGAGEMENT_LEVELS[0]
        if score <= thresholds[1]:
            return ENGAGEMENT_LEVELS[1]
        if score <= thresholds[2]:
            return ENGAGEMENT_LEVELS[2]
        return ENGAGEMENT_LEVELS[3]

    levels = proxy_scores.apply(_level)
    return levels, thresholds


def _distance_from_boundary_score(proxy_scores: pd.Series, thresholds: list[float]) -> pd.Series:
    """0 = sitting right on a level boundary (least confident), 1 = far from any boundary."""
    span = max(thresholds[-1] - thresholds[0], 1e-9)

    def _dist(score: float) -> float:
        if pd.isna(score):
            return float("nan")
        nearest = min(abs(score - t) for t in thresholds)
        return min(1.0, nearest / (span / 2))

    return proxy_scores.apply(_dist)


def compute_confidence(df: pd.DataFrame, proxy_scores: pd.Series, thresholds: list[float]) -> pd.Series:
    # 1. Data availability -- untrustworthy proxy if barely any real detections underlie it.
    detection_cols = [c for c in ("fk_detection_rate", "gt_robot_detection_rate") if c in df.columns]
    if detection_cols:
        availability = df[detection_cols].mean(axis=1, skipna=True).fillna(0.0)
    else:
        availability = pd.Series(1.0, index=df.index)

    # 2. Distance from nearest level boundary -- segments right on a threshold flip easiest.
    boundary_confidence = _distance_from_boundary_score(proxy_scores, thresholds)

    # 3. Cross-modality disagreement -- the design notes' "disagreement" active-learning lever.
    # Compared on percentile rank, not raw value -- same scale-mismatch reason
    # as compute_engagement_proxy (gt and fk are not on comparable scales).
    gt = df.get("gt_looking_at_robot_rate")
    fk = df.get("fk_gaze_target_camera_frac")
    if gt is not None and fk is not None:
        gt_rank = gt.rank(pct=True)
        fk_rank = fk.rank(pct=True)
        both = gt_rank.notna() & fk_rank.notna()
        disagreement = (gt_rank - fk_rank).abs()
        agreement_confidence = pd.Series(1.0, index=df.index)
        agreement_confidence[both] = 1.0 - disagreement[both].clip(0, 1)
    else:
        agreement_confidence = pd.Series(1.0, index=df.index)

    combined = availability.fillna(0.0) * boundary_confidence.fillna(0.5) * agreement_confidence.fillna(1.0)
    return combined


def label_dataframe(df: pd.DataFrame, review_fraction: float = REVIEW_FRACTION) -> pd.DataFrame:
    proxy = compute_engagement_proxy(df)
    levels, thresholds = assign_engagement_level(proxy)
    confidence = compute_confidence(df, proxy, thresholds)

    out = df.copy()
    out["engagement_proxy_score"] = proxy
    out["engagement_level"] = levels
    out["engagement_confidence"] = confidence

    valid_conf = confidence.dropna()
    cutoff = valid_conf.quantile(review_fraction) if not valid_conf.empty else 0.0
    out["needs_review"] = confidence.notna() & (confidence <= cutoff)
    out["needs_review"] = out["needs_review"] | confidence.isna()

    out.attrs["engagement_thresholds"] = thresholds
    return out
