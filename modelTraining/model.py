"""CAMEO shared-trunk model: per-camera-type branches -> graph-attention
group module -> shared trunk -> engagement head (trained) + behavior head
(architected, not trained -- Option B in docs/design-notes.md, behavior labels don't
exist yet).

Graphs are tiny (1-3 nodes, confirmed empirically on 03_20) and never mix
rgb/cam2 within one segment, so this processes one graph at a time (no
padding/masking machinery) -- simpler and correct at this data scale
(258 graphs total), not a shortcut that costs anything real here.

The graph-attention module and attention-pooled group representation are
the load-bearing novelty piece (per docs/design-notes.md: flat pooling would forfeit
the group-relational claim) -- edge features enter as an additive bias on
learned attention scores, not just as extra node features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RGB_IN_DIM = 124
# 11 gaze/position aggregates (build_graph_dataset.CAM2_NODE_FEATURE_COLS) +
# 19 Cam_2-native OpenFace 3.0 facial features (build_graph_dataset.
# OPENFACE_NODE_FEATURE_COLS: 8 AU + 8 emotion distribution + 2 gaze + 1
# availability flag), added 2026-07-23 -- see assemble_dataset.py's
# ALL_CAM2_NODE_FEATURE_COLS for the exact concatenation order.
CAM2_IN_DIM = 11 + 19
EMBED_DIM = 32
# 4th channel added 2026-07-23: attending_to_speaker (is node i's gaze on
# node j while j is talking, from build_graph_dataset.build_cam2_edges'
# mouth-motion-weighted containment) -- a richer joint-attention/turn-
# taking signal than plain mutual-gaze/proximity, testing whether the
# graph module needs a more behaviorally-grounded edge to show any benefit.
EDGE_DIM = 4
N_HEADS = 4
N_ENGAGEMENT_CLASSES = 4
N_BEHAVIOR_CLASSES = 6  # maintain/verbal-re-engage/non-verbal-re-engage/repair/escalate/disengage, per docs/design-notes.md

# Per-recording-date embedding (added 2026-08-01), for the pooled
# 03_20+03_26+05_15 continuous-scale run. Diagnosed via LOSO-checkpoint
# overlays + a pooled-dataset audit that the 3 worst-bias folds (03_20 s1,
# 03_20 s4, 03_26 s2) were confidently WRONG for tens of seconds at a time
# during genuine, unambiguous engagement states (not label noise, not the
# graph module -- no_graph showed the identical error -- and not fixable by
# post-hoc temporal smoothing, which left both bias and MAE unchanged since
# the error is a real wrong mean, not frame-to-frame jitter). The one
# surviving, evidenced explanation: 05_15's raw mean AU activation is ~2x
# 03_20/03_26's even after per-(source,dataset) normalization (see
# normalize.py's 2026-07-28 fix) -- normalization only equalizes scale, it
# can't fix a genuinely different features-to-engagement mapping across
# recording setups/lighting/populations. A fixed per-session output bias
# was ruled out earlier (error sign/magnitude varies WITHIN a single
# session, so a constant correction would break already-correct stretches
# to fix wrong ones) -- this embeds dataset identity into the INPUT instead,
# letting the shared per-node representation learn a different decision
# function per recording date rather than a single additive correction.
DATASET_VOCAB = ["03_20", "03_26", "05_15"]


class Branch(nn.Module):
    """One small MLP per camera-type feature space, projecting into the
    shared per-node embedding space the design notes' 'individual representation'
    step describes.

    Temporal GRU added 2026-07-30: the frame-level continuous pipeline
    (build_continuous_dataset.py) now feeds each node a (WINDOW_SIZE, in_dim)
    sequence -- that person's own feature history over the preceding ~3s --
    instead of one instant, per the literature (MultiMediate/DA-Mamba,
    DAiSEE-based studies) universally modeling engagement as evolving over a
    window of frames, not from a single instant. The per-step MLP (`net`)
    still projects each step into embed_dim exactly as before; a GRU then
    consumes that step sequence and its FINAL hidden state becomes the
    node's embedding, so downstream graph attention/pooling/heads see one
    temporally-aware vector per node, unchanged in shape from before.

    Backward compatible with the older segment-level pipeline
    (build_graph_dataset.py/train.py), which still passes single-instant,
    non-windowed features as a plain (n, in_dim) tensor: forward() checks
    the input's ndim and skips the GRU entirely for 2D input, running the
    exact same code path as before this change. Only 3D (n, W, in_dim)
    input (the new windowed continuous pipeline) exercises the GRU."""

    def __init__(self, in_dim: int, embed_dim: int = EMBED_DIM, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.temporal_gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.net(x)  # (n, in_dim) -> (n, embed_dim), unchanged single-instant path
        n, w, in_dim = x.shape
        step_embeds = self.net(x.reshape(n * w, in_dim)).reshape(n, w, -1)  # (n, W, embed_dim)
        _, h_n = self.temporal_gru(step_embeds)
        return h_n.squeeze(0)  # (n, embed_dim) -- final hidden state per node


class GraphAttentionLayer(nn.Module):
    """Dense multi-head self-attention over a graph's nodes, with edge
    features entering as an additive bias on the attention logits -- so the
    relational structure (gaze/proximity/mutual-gaze) actually shapes which
    nodes attend to which, not just extra unused input."""

    def __init__(self, embed_dim: int = EMBED_DIM, edge_dim: int = EDGE_DIM, n_heads: int = N_HEADS):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_bias_proj = nn.Linear(edge_dim, n_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, node_embeds: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        n = node_embeds.shape[0]
        q = self.q_proj(node_embeds).view(n, self.n_heads, self.head_dim)
        k = self.k_proj(node_embeds).view(n, self.n_heads, self.head_dim)
        v = self.v_proj(node_embeds).view(n, self.n_heads, self.head_dim)

        scores = torch.einsum("ihd,jhd->hij", q, k) / (self.head_dim ** 0.5)  # (heads, N, N)
        if n > 1:
            edge_bias = self.edge_bias_proj(edge_features).permute(2, 0, 1)  # (heads, N, N)
            scores = scores + edge_bias
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(n, -1)
        out = self.out_proj(out)
        return self.norm(node_embeds + out)


class SimpleRelationalLayer(nn.Module):
    """Alternative to GraphAttentionLayer, testing whether full multi-head
    self-attention is overkill for graphs this small (1-3 nodes,
    confirmed empirically -- attention is built to select among many
    competitors, and there's nothing to select among with 1-3).

    Edge features (gaze rate / proximity / mutual gaze) drive the mixing
    weight DIRECTLY (via one small linear projection to a scalar, then
    softmax over neighbors) -- not added as a bias competing against a
    separately-learned content-based attention score the way
    GraphAttentionLayer does it. Far fewer parameters (one edge_dim->1
    projection + one value projection, vs four embed_dim x embed_dim
    projections + a separate edge bias). Tests the hypothesis (2026-07-22)
    that the heavyweight attention formulation, not the relational idea
    itself, is why 'full' hasn't beaten 'no_graph' in any run so far."""

    def __init__(self, embed_dim: int = EMBED_DIM, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.edge_weight_proj = nn.Linear(edge_dim, 1)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, node_embeds: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        n = node_embeds.shape[0]
        if n == 1:
            return node_embeds  # no one to relate to

        v = self.value_proj(node_embeds)  # (n, embed_dim)
        raw_weights = self.edge_weight_proj(edge_features).squeeze(-1)  # (n, n), edge (i,j) -> weight
        self_mask = torch.eye(n, dtype=torch.bool, device=node_embeds.device)
        raw_weights = raw_weights.masked_fill(self_mask, float("-inf"))  # no self-loop, residual handles that
        weights = F.softmax(raw_weights, dim=-1)  # normalize over neighbors j, per row i
        mixed = weights @ v  # (n, embed_dim)
        return self.norm(node_embeds + mixed)


class GroupPooling(nn.Module):
    """Attention-pooled group representation -- deliberately not mean/flat
    pooling, since the design notes are explicit that flat pooling forfeits the
    group-relational novelty claim. mode='mean' exists only for the
    ablation ladder (to demonstrate attention pooling is doing real work,
    not just "a graph was added")."""

    def __init__(self, embed_dim: int = EMBED_DIM, mode: str = "attention"):
        super().__init__()
        assert mode in ("attention", "mean")
        self.mode = mode
        self.query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.scale = embed_dim ** 0.5

    def forward(self, node_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "mean":
            n = node_embeds.shape[0]
            weights = torch.full((n,), 1.0 / n, device=node_embeds.device)
            return node_embeds.mean(dim=0), weights
        scores = (node_embeds @ self.query) / self.scale  # (N,)
        weights = F.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * node_embeds).sum(dim=0)
        return pooled, weights


class CAMEOModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_graph_layers: int = 2,
        pooling: str = "attention",
        use_graph: bool = True,
        graph_type: str = "attention",
        dropout: float = 0.2,
        head_type: str = "classification",
        cam2_in_dim: int = CAM2_IN_DIM,
        edge_dim: int = EDGE_DIM,
        use_dataset_embed: bool = False,
    ):
        """pooling/use_graph exist for the ablation ladder (see train.py):
        use_graph=False skips message passing entirely (per-person-only
        baseline); pooling='mean' replaces attention pooling with a flat
        average (tests whether attention pooling is doing real work).
        graph_type='simple' swaps GraphAttentionLayer for
        SimpleRelationalLayer -- tests whether full multi-head attention
        is the wrong tool at this graph size (1-3 nodes) vs a lighter,
        edge-weight-direct mixing layer. dropout added 2026-07-23 as a
        regularization lever -- sanity_check.py hits ~100% train accuracy
        easily while held-out accuracy sits around 0.42-0.45, a classic
        overfitting signature, worth tuning directly rather than assuming
        the fixed 0.2 default is right. cam2_in_dim overrides the default
        segment-level 30-dim cam2 feature width -- the frame-level
        continuous-scale dataset (build_continuous_dataset.py) uses a
        different, non-aggregated 29-dim per-frame feature vector, since
        there's no 3s window to compute mean/std aggregates over.
        edge_dim overrides the default 4-channel edge width -- the
        frame-level dataset added a 5th channel 2026-07-25 (neighbor's own
        engagement ~1s earlier, a peer-contagion/shared-context signal,
        see build_continuous_dataset.py's LAG_SEC), which the original
        4 gaze-direction-only channels never encoded."""
        super().__init__()
        self.use_graph = use_graph
        self.rgb_branch = Branch(RGB_IN_DIM, embed_dim, dropout=dropout)
        self.cam2_branch = Branch(cam2_in_dim, embed_dim, dropout=dropout)
        self.use_dataset_embed = use_dataset_embed
        if use_dataset_embed:
            self.dataset_embed = nn.Embedding(len(DATASET_VOCAB), embed_dim)
        assert graph_type in ("attention", "simple")
        layer_cls = GraphAttentionLayer if graph_type == "attention" else SimpleRelationalLayer
        self.graph_layers = nn.ModuleList([layer_cls(embed_dim, edge_dim=edge_dim) for _ in range(n_graph_layers)])
        self.group_pool = GroupPooling(embed_dim, mode=pooling)

        trunk_dim = embed_dim * 2  # per-node embedding concat pooled group representation
        # 4-way classification head, trained with cross-entropy as before.
        # Plain MSE regression on a single continuous output was tried
        # (2026-07-23) and rejected: a regularization sweep (wd 1e-4 to
        # 5e-3, dropout 0.1 to 0.4) plus removing the output-bounding
        # sigmoid both left predictions compressed toward the center,
        # regardless of setting -- this ruled out regularization/activation
        # choice as the cause. It's an inherent property of squared-error
        # loss under real predictive uncertainty (the MSE-optimal prediction
        # IS the conditional mean, so the model hedges when unsure) that
        # cross-entropy doesn't have, since it must commit to one discrete
        # class regardless of confidence -- which is why the classifier was
        # good at the extremes in the first place. The continuous score
        # this project actually wants (see train.py) is instead derived at
        # evaluation time as the expected value under this head's own
        # softmax probabilities (score = sum_k P(class=k) * anchor_k) --
        # keeps cross-entropy's confident-extreme behavior, while still
        # producing a continuous, tolerance-band-evaluable score, since
        # probability mass naturally spreads across adjacent classes
        # exactly when the model is genuinely torn between them.
        # head_type="regression" added 2026-07-24 for the continuous-scale
        # (per-frame, not per-3s-segment) 05_15 experiment -- single
        # sigmoid*100-bounded scalar per node, trained with a tolerance-band
        # ("epsilon-insensitive") loss in train.py rather than plain MSE.
        # Plain MSE regression (see above) was rejected for mean-collapse;
        # this is a genuinely different loss shape (zero gradient once a
        # prediction is already within the tolerance band, so the model
        # isn't rewarded for creeping even closer to the exact target the
        # way squared error always pushes it to) -- worth testing on its
        # own merits rather than assuming the earlier rejection still holds.
        assert head_type in ("classification", "regression", "ordinal")
        self.head_type = head_type
        # "ordinal" (CORN, added 2026-07-30): K classes -> K-1 rank-consistent
        # threshold logits, each modeling the conditional P(y>k | y>k-1) --
        # targets the confirmed adjacent-class-confusion collapse seen when
        # pooling multiple recording dates (plain softmax has no notion that
        # "medium" sits between "low" and "high", so extreme classes win as
        # the pool gets more heterogeneous). No forward-pass change needed
        # beyond out_dim -- CORN's loss/eval (train_continuous.py's
        # corn_loss/corn_probas_from_logits/corn_label_from_logits) apply
        # sigmoid internally, same as classification leaves its logits raw.
        if head_type == "classification":
            out_dim = N_ENGAGEMENT_CLASSES
        elif head_type == "ordinal":
            out_dim = N_ENGAGEMENT_CLASSES - 1
        else:
            out_dim = 1
        self.engagement_head = nn.Sequential(
            nn.Linear(trunk_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, out_dim)
        )
        # Present in the forward pass so the shared-trunk claim is real and
        # inspectable, but excluded from the loss entirely (see train.py's
        # train_behavior_head=False) -- Option B: no behavior labels exist
        # yet, this head is architected/designed, not trained.
        self.behavior_head = nn.Sequential(
            nn.Linear(trunk_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, N_BEHAVIOR_CLASSES)
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, source: str, dataset: str | None = None):
        branch = self.rgb_branch if source == "rgb" else self.cam2_branch
        node_embeds = branch(node_features)

        if self.use_dataset_embed and dataset is not None:
            # dataset=None is a deliberate backoff (2026-08-01), not a missing
            # argument -- see train_continuous.py's EMBED_ELIGIBLE_DATASETS.
            # Verified empirically: giving 03_20 (only 2 sessions total) its
            # own learned embedding made both its LOSO folds WORSE (mae
            # 26.84->33.79 and 35.73->38.16), because with only one sibling
            # session left in training per fold, the embedding just learns
            # that one session's own average rather than a genuine
            # per-recording-date signal -- confirmed the same mechanism
            # HELPED 03_26 s2 (mae 34.56->31.34), which has 5 siblings to
            # learn from. Callers pass dataset=None for datasets without
            # enough sessions to support a real embedding, which skips this
            # block entirely -- identical to use_dataset_embed=False for
            # that specific item.
            if dataset not in DATASET_VOCAB:
                raise ValueError(f"use_dataset_embed=True requires dataset in {DATASET_VOCAB} or None, got {dataset!r}")
            idx = torch.tensor(DATASET_VOCAB.index(dataset), device=node_embeds.device)
            node_embeds = node_embeds + self.dataset_embed(idx)  # broadcasts over all nodes in this graph

        if self.use_graph:
            for layer in self.graph_layers:
                node_embeds = layer(node_embeds, edge_features)

        group_repr, attn_weights = self.group_pool(node_embeds)
        n = node_embeds.shape[0]
        shared = torch.cat([node_embeds, group_repr.unsqueeze(0).expand(n, -1)], dim=-1)

        engagement_out = self.engagement_head(shared)
        if self.head_type == "regression":
            engagement_out = torch.sigmoid(engagement_out.squeeze(-1)) * 100.0  # (n,), bounded 0-100
        behavior_logits = self.behavior_head(shared)
        return engagement_out, behavior_logits, attn_weights
