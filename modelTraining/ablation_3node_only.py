"""Re-run the ablation ladder restricted to 3-node segments only -- 1-node
segments have no relational structure at all (trivial self-attention) and
2-node segments have minimal structure, so if the graph module's value is
being diluted by segments where it can't do anything useful, this isolates
the case where it has the most to work with."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train import LEVELS, leave_one_session_out

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    data = torch.load(PROJECT_ROOT / "modelTraining" / "graph_dataset_03_20.pt", weights_only=False)
    three_node = [d for d in data if d["node_features"].shape[0] == 3]
    print(f"{len(three_node)} three-node graphs (out of {len(data)} total)")

    variants = ["full", "proximity_only", "mean_pool", "no_graph"]
    for variant in variants:
        result = leave_one_session_out(three_node, variant, "cpu")
        print(f"{variant:20s} mean_acc={result['mean_accuracy']:.3f}  mean_macro_f1={result['mean_macro_f1']:.3f}")
        for f in result["fold_results"]:
            print(f"    session {f['session']}: n={f['n_test']}, acc={f['accuracy']:.3f}, macro_f1={f['macro_f1']:.3f}")
