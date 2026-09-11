"""Drives a (virtual) Furhat from CAMEO's per-frame engagement predictions, so
the video demo can show a robot reacting to a real recorded session rather than
a plot of the same numbers.

IMPORTANT -- what this is and is not. The behaviour head described in the report
is UNTRAINED: no behaviour labels were annotated, so nothing here is a learned
policy. The mapping from engagement to robot action below is a hand-written
rule sitting on top of the model's engagement estimate. It exists to make the
downstream half of the architecture visible, not to claim it was learned. Say
so on screen if this is used in the submitted video.

Predictions are read from a completed leave-one-session-out sweep, so the
engagement values driving the robot are genuine out-of-sample predictions for
a session the model never trained on -- the same slicing that
render_prediction_overlay.py uses, kept identical so the robot and the overlay
video agree frame for frame.

Usage:
    # no SDK needed -- prints the action timeline it WOULD execute
    modelTraining/.venv/bin/python3 modelTraining/furhat_demo_driver.py \
        --results modelTraining/loso_results_3sess_noembed.pt \
        --variant-key full_classification --dataset 05_15 --session 2 \
        --start-sec 42.8 --end-sec 117.8 --dry-run

    # drive a virtual Furhat running in the SDK on this machine
    ... --host 127.0.0.1

    # from WSL, pointing at the SDK on the Windows host
    ... --host $(ip route show default | awk '{print $3}')
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent

# Where each seat sits relative to the robot, in Furhat's attention coordinate
# frame (x right, y up, z forward, metres). The three participants sit in an
# arc facing the robot's position, so these are a straight left/centre/right
# split at roughly conversational distance. Adjust if the demo footage is
# framed differently.
SEAT_LOCATIONS = {
    "A": (-0.9, 0.0, 1.2),
    "B": (0.0, 0.0, 1.2),
    "C": (0.9, 0.0, 1.2),
}

# Where the robot looks when it is addressing the group rather than a person.
# Deliberately NOT a seat: B sits dead centre, so a neutral pose at (0, 0, 1.2)
# makes "turned to address B" and "went back to watching the room" the same
# picture, and the viewer cannot tell an intervention happened. Lifting and
# lengthening the gaze keeps the two visibly distinct.
NEUTRAL_LOCATION = (0.0, 0.25, 2.2)

# How long the robot holds its gaze on someone it has just addressed. A line
# takes roughly three to four seconds to speak, so anything shorter turns the
# head away mid-sentence.
SPEECH_HOLD_SEC = 5.0

# Gesture names are the Furhat built-ins; check them against the gesture list in
# your SDK version before recording, since an unknown name is silently ignored
# and the head will simply not move.
GESTURE_REENGAGE = "BrowRaise"
GESTURE_ACKNOWLEDGE = "Nod"

REENGAGE_LINES = [
    "{name}, did you have any questions about that?",
    "{name}, does that all make sense so far?",
    "Let me check in with you, {name}.",
]

# A person must sit below this predicted engagement continuously for this long
# before the robot intervenes, and the robot then stays quiet for the cooldown
# regardless of what anyone does. Without the dwell it fires on single-frame
# dips; without the cooldown it talks over itself continuously in the stretches
# where the whole group is disengaged.
DISENGAGED_BELOW = 40.0
DWELL_SEC = 3.0
COOLDOWN_SEC = 12.0

# Attention hysteresis. For long stretches all three predictions sit within a
# point or two of each other (the model couples participants more tightly than
# the annotators did), so a plain argmax re-targets the head several times a
# second and reads on camera as a malfunction rather than as attention. A new
# person must therefore beat the current focus by ATTEND_MARGIN points, and the
# head holds each target at least ATTEND_MIN_HOLD_SEC.
ATTEND_MARGIN = 8.0
ATTEND_MIN_HOLD_SEC = 2.0

# While the whole group stays engaged the rule above correctly does nothing,
# which is right but leaves the robot frozen for a minute at a time. A periodic
# acknowledgement toward the current focus is the "maintain" class of the
# behaviour taxonomy, and keeps the head alive without inventing an
# intervention the engagement estimate does not support.
MAINTAIN_EVERY_SEC = 15.0


def load_predictions(results_path: Path, variant_key: str, dataset: str, session: int,
                     fps: float) -> list[dict]:
    """Per-frame per-participant predictions for one fold, as a time-ordered
    list of {t, scores{letter: value}}. Mirrors render_prediction_overlay's
    render_from_results slicing exactly: folds are concatenated in fold_results
    order, and each fold's nodes follow the graph list's own order."""
    results = torch.load(results_path, weights_only=False)[variant_key]
    offset = 0
    n_test = None
    for fr in results["fold_results"]:
        if fr["dataset"] == dataset and fr["session"] == session:
            n_test = fr["n_test"]
            break
        offset += fr["n_test"]
    if n_test is None:
        raise SystemExit(f"no fold for {dataset} session {session} in {results_path} [{variant_key}]")
    preds = results["all_preds"][offset:offset + n_test]

    graphs = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{dataset}_continuous.pt",
                        weights_only=False)
    session_graphs = sorted((g for g in graphs if g["session"] == session),
                            key=lambda g: g["frame_index"])

    by_frame: dict[int, dict[str, float]] = defaultdict(dict)
    cursor = 0
    for g in session_graphs:
        for i in range(len(g["labels"])):
            by_frame[g["frame_index"]][g["participants"][i]] = float(preds[cursor + i])
        cursor += len(g["labels"])
    if cursor != n_test:
        raise SystemExit(f"node count mismatch: graphs contributed {cursor}, results had {n_test}")

    return [{"t": f / fps, "scores": by_frame[f]} for f in sorted(by_frame)]


def plan_actions(timeline: list[dict], start_sec: float | None, end_sec: float | None) -> list[dict]:
    """Turn the engagement timeline into a list of timestamped robot actions.

    This is the hand-written policy standing in for the untrained behaviour
    head. One rule: whoever has been below DISENGAGED_BELOW the longest, once
    they pass DWELL_SEC, gets attended to and addressed; otherwise the robot
    attends whoever is currently most engaged. Deliberately simple, because the
    point of the demo is that the ENGAGEMENT ESTIMATE is doing the work."""
    actions: list[dict] = []
    below_since: dict[str, float | None] = {}
    last_intervention = -1e9
    last_attended: str | None = None
    last_attend_t = -1e9
    last_maintain = -1e9

    for frame in timeline:
        t, scores = frame["t"], frame["scores"]
        if start_sec is not None and t < start_sec:
            continue
        if end_sec is not None and t > end_sec:
            break
        if not scores:
            continue

        for letter, value in scores.items():
            if value < DISENGAGED_BELOW:
                below_since.setdefault(letter, t)
            else:
                below_since[letter] = None

        # longest continuously-disengaged person that clears the dwell
        candidates = [(t - since, letter) for letter, since in below_since.items()
                      if since is not None and t - since >= DWELL_SEC]
        candidates.sort(reverse=True)

        if candidates and t - last_intervention >= COOLDOWN_SEC:
            dwell, letter = candidates[0]
            actions.append({
                "t": round(t, 2), "kind": "reengage", "target": letter,
                "score": round(scores.get(letter, float("nan")), 1),
                "dwell_sec": round(dwell, 1),
                "location": SEAT_LOCATIONS.get(letter),
                "gesture": GESTURE_REENGAGE,
                "say": REENGAGE_LINES[len(actions) % len(REENGAGE_LINES)].format(name=f"participant {letter}"),
            })
            last_intervention = t
            last_attended = letter
            last_attend_t = t
            below_since[letter] = None  # treat as addressed; re-arm from here
            continue

        # nobody needs rescuing: look at whoever is most engaged, subject to the
        # hysteresis above so the head settles instead of oscillating
        focus, focus_score = max(scores.items(), key=lambda kv: kv[1])
        if focus != last_attended and t - last_attend_t >= ATTEND_MIN_HOLD_SEC:
            current = scores.get(last_attended) if last_attended else None
            if current is None or focus_score - current >= ATTEND_MARGIN:
                actions.append({
                    "t": round(t, 2), "kind": "attend", "target": focus,
                    "score": round(focus_score, 1),
                    "location": SEAT_LOCATIONS.get(focus),
                    "gesture": GESTURE_ACKNOWLEDGE if not actions else None,
                    "say": None,
                })
                last_attended = focus
                last_attend_t = t
                last_maintain = t
                continue

        if (last_attended is not None
                and t - max(last_attend_t, last_maintain, last_intervention) >= MAINTAIN_EVERY_SEC):
            actions.append({
                "t": round(t, 2), "kind": "maintain", "target": last_attended,
                "score": round(scores.get(last_attended, float("nan")), 1),
                "location": SEAT_LOCATIONS.get(last_attended),
                "gesture": GESTURE_ACKNOWLEDGE,
                "say": None,
            })
            last_maintain = t

    return actions


def choreograph(actions: list[dict], start_sec: float | None) -> list[dict]:
    """Reduce the raw plan to something legible on camera.

    The raw plan re-targets the head whenever the argmax moves, which is
    faithful to the engagement estimate but reads as aimless: the robot turns
    to people nothing is happening with, and turns away from people it is
    mid-sentence with. For the video, only the interventions carry meaning, so
    plain 'attend' moves are dropped, every intervention is held for the length
    of its line, and the robot then returns to watching the group."""
    out: list[dict] = []

    opening = start_sec if start_sec is not None else (actions[0]["t"] if actions else 0.0)
    out.append({"t": round(opening, 2), "kind": "watch", "target": "group", "score": None,
                "location": NEUTRAL_LOCATION, "gesture": GESTURE_ACKNOWLEDGE, "say": None})

    for action in actions:
        if action["kind"] == "reengage":
            out.append(action)
            out.append({"t": round(action["t"] + SPEECH_HOLD_SEC, 2), "kind": "watch",
                        "target": "group", "score": None, "location": NEUTRAL_LOCATION,
                        "gesture": None, "say": None})
        elif action["kind"] == "maintain":
            # nod while facing the group, not while staring at one person
            out.append({**action, "target": "group", "location": NEUTRAL_LOCATION})

    out.sort(key=lambda a: a["t"])

    # A return-to-group immediately before the next intervention is a wasted
    # move: the head would swing back only to swing straight out again.
    tidied: list[dict] = []
    for i, action in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if action["kind"] == "watch" and nxt is not None and nxt["t"] - action["t"] < 2.0:
            continue
        tidied.append(action)
    return tidied


def execute(actions: list[dict], host: str, auth_key: str | None, speed: float,
            language: str = "en-GB") -> None:
    """Replay the planned actions against a live (virtual) Furhat in real time."""
    try:
        from furhat_realtime_api import FurhatClient
    except ImportError:
        raise SystemExit("pip install furhat_realtime_api (or pass --dry-run)")
    from furhat_play_actions import configure_voice

    furhat = FurhatClient(host, auth_key=auth_key)
    try:
        furhat.connect()
    except Exception as exc:  # noqa: BLE001 -- surface the SDK's own message
        raise SystemExit(
            f"could not connect to Furhat at {host}: {type(exc).__name__} {exc}\n"
            "A timeout here usually means the connection is not local: the Realtime API answers\n"
            "keyless auth only for localhost, and ignores it from other hosts. Either pass\n"
            "--auth-key, or plan with --save-actions here and replay on the SDK machine with\n"
            "furhat_play_actions.py.")

    configure_voice(furhat, language, gender=None)

    t0 = time.monotonic()
    origin = actions[0]["t"] if actions else 0.0
    for action in actions:
        target_wall = (action["t"] - origin) / speed
        while (elapsed := time.monotonic() - t0) < target_wall:
            time.sleep(min(0.02, target_wall - elapsed))

        if action["location"]:
            x, y, z = action["location"]
            furhat.request_attend_location(x, y, z)
        if action["gesture"]:
            furhat.request_gesture_start(action["gesture"])
        if action["say"]:
            # wait=False so speech does not block the next scheduled action and
            # drift the demo out of sync with the video playing alongside it
            furhat.request_speak_text(action["say"], wait=False)
        print(f"[{action['t']:7.2f}s] {action['kind']:8s} {action['target']} "
              f"(score {action['score']})" + (f"  say: {action['say']!r}" if action["say"] else ""))

    furhat.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="modelTraining/loso_results_3sess_noembed.pt")
    parser.add_argument("--variant-key", default="full_classification")
    parser.add_argument("--dataset", default="05_15")
    parser.add_argument("--session", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="footage fps for this date -- 05_14 is 60, everything else 30")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--auth-key", default=None)
    parser.add_argument("--speed", type=float, default=1.0, help=">1 replays faster than real time")
    parser.add_argument("--language", default="en-GB", help="voice language; the SDK defaults to Swedish")
    parser.add_argument("--style", choices=["demo", "raw"], default="demo",
                        help="demo: interventions only, held and returned to group (legible on camera). "
                             "raw: every attention change the policy makes")
    parser.add_argument("--dry-run", action="store_true", help="print the timeline, touch no robot")
    parser.add_argument("--save-actions", default=None,
                        help="write the planned actions to JSON, for rendering a caption "
                             "track over the session video if the SDK is unavailable")
    args = parser.parse_args()

    timeline = load_predictions(Path(args.results), args.variant_key, args.dataset, args.session, args.fps)
    print(f"{len(timeline)} predicted frames, {timeline[0]['t']:.1f}s to {timeline[-1]['t']:.1f}s", file=sys.stderr)

    actions = plan_actions(timeline, args.start_sec, args.end_sec)
    if args.style == "demo":
        actions = choreograph(actions, args.start_sec)
    print(f"planned {len(actions)} actions "
          f"({sum(a['kind'] == 'reengage' for a in actions)} re-engagements), style={args.style}",
          file=sys.stderr)

    if args.save_actions:
        Path(args.save_actions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_actions).write_text(json.dumps(actions, indent=2))
        print(f"wrote {args.save_actions}", file=sys.stderr)

    if args.dry_run:
        for a in actions:
            line = f"[{a['t']:7.2f}s] {a['kind']:8s} {a['target']} (score {a['score']})"
            if a["kind"] == "reengage":
                line += f" after {a['dwell_sec']}s disengaged -> {a['say']!r}"
            print(line)
        return

    execute(actions, args.host, args.auth_key, args.speed, args.language)


if __name__ == "__main__":
    main()
