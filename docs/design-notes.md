# CAMEO design notes

Architecture and methodology decisions taken before implementation, kept here because
several source files cite them by name. Where a decision was later revised by evidence,
the revision is recorded alongside it rather than overwriting it.

## Project summary

CAMEO (Context-Aware Multimodal Engagement-Oriented interaction model) estimates
engagement for every co-present participant jointly, and was designed to unify
engagement detection and robot behaviour generation in a single model rather than
treating them as separate perception-then-policy stages. Target platform: Furhat
(physical and the Virtual Furhat simulator). Use case: healthcare reception, both
individual and group interactions.

## Positioning against the literature

- Engagement-detection work identifies context-specific datasets and temporal dynamics
  as open gaps, which the purpose-built corpus here addresses directly.
- Group-HRI work explicitly flags subgroup and interpersonal-relationship detection as
  underdeveloped. The group module targets this, but only if it is a relational or
  structured representation (attention over a graph of individuals) rather than a flat
  pooled average of per-person scores.
- Cross-modal attention fusion for engagement is already a mature area (MultiMediate,
  dialogue-aware transformers, cross-attention with affective embeddings). It is standard
  machinery here, not a novelty claim.
- Engagement perception and behaviour generation are consistently treated as separate
  systems in prior work: a scalar engagement score handed to a downstream or hand-authored
  policy. The architectural claim here is a shared trunk, one representation both tasks
  read from and both losses shape.
- Robot-receptionist work (Gunson et al.; Moujahid, Hastie and Lemon) focuses on
  multi-party dialogue and turn-taking rather than continuous affective engagement,
  which supports the healthcare-reception plus affective-engagement combination as
  underexplored.

A later literature check narrowed the novelty claim. Both of the closest papers
(Zhang 2022, HRI-GAT; DS-HGCN 2025, hypergraph) were read in full: neither propagates an
engagement value along an edge, neither is directed-pairwise, and neither is lagged.
A generic graph-relational claim is therefore **not** novel; the lagged peer-engagement
edge mechanism is the defensible part.

## Architecture: shared trunk

```
[Facial]  [Prosody]  [Pose]  [Text]        <- per-modality encoders
      \       |         |      /
       Cross-modal attention fusion         <- standard mechanism, not the novelty
              /              \
Individual representation   Group affective state
  (per-person fused state)   (cross-person relational model: graph/relational,
                              not flat pooling)
              \              /
         Shared representation              <- the key architectural claim
         (single trunk, joint loss)
              /              \
   Engagement head      Behaviour policy head
   (predicts engagement (selects robot response:
    score)                maintain / verbal re-engage /
                          non-verbal re-engage / repair /
                          escalate / disengage gracefully)
```

For the unified-model claim to hold, behaviour-loss gradients must flow back into the
shared trunk, not merely read from it.

### Training strategy

- **Option A, fully joint from step one.** `L_total = L_engagement + lambda * L_behaviour`,
  both losses shaping the trunk from the first gradient step. Strongest claim, but requires
  behaviour labels to exist alongside engagement labels from the start.
- **Option B, pretrain then joint fine-tune.** Train trunk and engagement head first on
  annotated engagement data, then unfreeze and continue with both losses. Still satisfies
  "shared representation shaped by both tasks", just not from the first step.
- **Option C, frozen trunk, separate behaviour training.** A standard two-stage pipeline
  with a shared feature extractor. Honest framing if this is what happens would be
  "modular pipeline with shared features", not a unified model.

**Outcome: Option B, with the second phase not reached.** No behaviour labels were
annotated, so the behaviour head is architected and wired into the model but never
trained. `train.py` keeps it out of the loss and raises on any attempt to train it.
Anything in the demo that maps engagement to a robot action is a hand-written rule
sitting on top of the engagement estimate, not a learned policy, and is labelled as
such in `modelTraining/furhat_demo_driver.py`.

## Annotation strategy

The recordings are multi-terabyte with thousands of frames per video, so the original
plan was segment-level labelling:

1. **Segment, don't frame-label.** Established engagement datasets (e.g. CMOSE) label
   short segments rather than individual frames, since engagement does not meaningfully
   change frame to frame.
2. **Sample a representative subset first**, across scenarios and individual versus group,
   rather than processing everything.
3. **Auto-label with existing tools, then human-verify** rather than labelling from
   scratch: off-the-shelf AU, gaze and pose detectors for a first pass.
4. **Active learning**: route only low-confidence or disagreement segments to human review.
5. **Report inter-rater reliability** on a human-verified gold subset.
6. **Behaviour labels ride on the same segments** as engagement labels, using a small
   closed taxonomy to keep the policy head a tractable classifier.

**Outcome: superseded by continuous annotation.** Segment-level labelling was measured to
be too coarse: 66 per cent of ten-second windows span more than one engagement level.
The corpus was therefore annotated continuously, frame by frame, for every participant
rather than at segment grain for one designated target. The auto-labelling heuristic in
`datasetPrep/engagement_heuristic.py` and the `auto_label_*` scripts are retained as the
first-pass tooling that this decision replaced.

## Validation strategy without guaranteed physical Furhat access

- **Virtual Furhat is a legitimate primary validation environment**, not a fallback. It is
  the officially supported development path, runs the same SDK and architecture, and
  mirrors gestures, gaze and facial animation.
- **Offline validation** against held-out annotated sessions requires no robot at all.
  This is what the reported results are: leave-one-session-out cross-validation.
- **Video-based playback studies** are an established HRI substitute for live deployment:
  show raters video of the robot executing chosen behaviours and collect subjective ratings.
- Framing: offline metric validation, then Virtual Furhat behavioural validation, with
  physical deployment stated explicitly as future work.
