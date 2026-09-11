"""Baseline engagement classifiers on the 03_20 manually-labeled pilot.

Two separate models, not one combined model -- track_id (rgb, sessions
1/2/5) and person_idx (cam2, sessions 3/4) rows have genuinely different
feature spaces (fk_* blendshapes/head-pose vs gtp_* gaze-only), and
session 5 lacks gt_* entirely (no Cam_2 coverage), so forcing everything
into one feature matrix would mean heavy imputation muddying what's
actually being tested.

Evaluation is leave-one-session-out (not a random split): each session is
held out in turn as the test fold, trained on the rest. This is the real
test of interest -- given how much the earlier EDA showed sessions vary
(match rate, feature coverage), a random split would let the model
partially memorize session-specific quirks rather than proving it
generalizes to an unseen session. Compared against a majority-class
baseline computed the same way (predict the training folds' most common
label), so "beats trivial guessing" has an actual number to clear.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
LEVELS = ["disengaged", "low", "medium", "high"]


def _leave_one_session_out(df: pd.DataFrame, feature_cols: list[str], label: str) -> None:
    # inout_score is never actually populated anywhere in gaze_target's raw output
    # (present in the CSV schema, but always blank) -- drop any feature that's
    # entirely empty rather than let it force-drop every row via dropna.
    feature_cols = [c for c in feature_cols if df[c].notna().any()]
    df = df.dropna(subset=feature_cols)

    sessions = sorted(df["session"].unique())
    rows = []
    all_importances = []

    for held_out in sessions:
        train = df[df["session"] != held_out]
        test = df[df["session"] == held_out]
        if test.empty or train.empty:
            continue

        X_train, y_train = train[feature_cols], train["human_engagement_level"]
        X_test, y_test = test[feature_cols], test["human_engagement_level"]

        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        majority = y_train.value_counts().idxmax()
        baseline_pred = [majority] * len(y_test)

        rows.append({
            "held_out_session": held_out,
            "n_test": len(test),
            "model_accuracy": accuracy_score(y_test, pred),
            "model_macro_f1": f1_score(y_test, pred, labels=LEVELS, average="macro", zero_division=0),
            "majority_accuracy": accuracy_score(y_test, baseline_pred),
            "majority_macro_f1": f1_score(y_test, baseline_pred, labels=LEVELS, average="macro", zero_division=0),
        })
        all_importances.append(clf.feature_importances_)

    result = pd.DataFrame(rows)
    print(f"=== {label}: leave-one-session-out ({len(feature_cols)} features, {len(df)} rows) ===")
    print(result.to_string(index=False))
    print(f"\nmean model accuracy: {result['model_accuracy'].mean():.3f}  "
          f"mean model macro-F1: {result['model_macro_f1'].mean():.3f}")
    print(f"mean majority-baseline accuracy: {result['majority_accuracy'].mean():.3f}  "
          f"mean majority-baseline macro-F1: {result['majority_macro_f1'].mean():.3f}")

    if all_importances:
        avg_importance = np.mean(all_importances, axis=0)
        top = pd.Series(avg_importance, index=feature_cols).sort_values(ascending=False).head(10)
        print("\ntop 10 features by average importance across folds:")
        print(top.to_string())
    print()


def main(dataset: str = "03_20") -> None:
    df = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "manual_labeled_dataset.parquet")
    matched = df[df["match_status"] == "matched"].copy()

    fk_cols = [c for c in matched.columns if c.startswith("fk_")]
    track_df = matched[matched["id_type"] == "track_id"].copy()
    _leave_one_session_out(track_df, fk_cols, "track_id / rgb sessions (facial_keypoints features)")

    gtp_cols = [c for c in matched.columns if c.startswith("gtp_")]
    gt_cols = [c for c in matched.columns if c.startswith("gt_")]
    person_df = matched[matched["id_type"] == "person_idx"].copy()
    _leave_one_session_out(person_df, gtp_cols + gt_cols, "person_idx / cam2 sessions (gaze_target features)")


if __name__ == "__main__":
    main()
