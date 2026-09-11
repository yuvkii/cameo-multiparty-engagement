"""Replays a saved CAMEO action timeline on a (virtual) Furhat.

Standalone on purpose: this imports NOTHING from the rest of the project and
does not need torch, so it can be run directly on the machine hosting the
Furhat SDK. That matters because the Realtime API answers keyless auth only
for local connections -- a client on another host (e.g. WSL talking to the
SDK on the Windows side) gets its request.auth silently ignored. Running the
planner in WSL, saving its output, and replaying it here sidesteps the whole
problem.

Produce the timeline first, in WSL:
    modelTraining/.venv/bin/python3 modelTraining/furhat_demo_driver.py \
        --start-sec 42.8 --end-sec 117.8 --dry-run \
        --save-actions outputs/demo/actions_05_15_s2.json

Then, on the machine running the SDK:
    pip install furhat_realtime_api
    python furhat_play_actions.py --actions actions_05_15_s2.json

The repo is reachable from Windows at \\\\wsl$\\<distro>\\home\\yuki\\individualProject,
so both files can be used in place without copying anything.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path


def _voice_entries(voice_status: dict) -> list[dict]:
    """The API has returned voice_list as plain id strings in some versions and
    as dicts in others; normalise both to dicts so selection logic below does
    not depend on which one this SDK ships."""
    entries = []
    for v in voice_status.get("voice_list", []):
        if isinstance(v, str):
            entries.append({"id": v, "name": v, "language": "", "gender": ""})
        elif isinstance(v, dict):
            entries.append({
                "id": v.get("id") or v.get("name") or "",
                "name": v.get("name") or v.get("id") or "",
                "language": v.get("language") or v.get("lang") or "",
                "gender": v.get("gender") or "",
            })
    return entries


def configure_voice(furhat, language: str, gender: str | None, verbose: bool = True) -> None:
    """Pick an English voice and report what was actually selected.

    The SDK defaults to a Swedish voice, and request_voice_config silently
    keeps the current voice if nothing matches, so this checks afterwards and
    says what it got rather than assuming the request took effect."""
    try:
        status = furhat.request_voice_status()
    except Exception as exc:  # noqa: BLE001
        print(f"could not read voice status ({exc}); leaving voice unchanged")
        return

    before = status.get("voice_id")
    entries = _voice_entries(status)

    exact = [v for v in entries if v["language"].lower().replace("_", "-") == language.lower()]
    family = [v for v in entries if v["language"].lower().startswith(language.split("-")[0].lower())]
    # Fall back to matching on the id when the API exposes no language field
    by_id = [v for v in entries if language.lower().replace("-", "") in v["id"].lower().replace("-", "").replace("_", "")]
    candidates = exact or family or by_id

    if gender:
        gendered = [v for v in candidates if v["gender"].lower() == gender.lower()]
        candidates = gendered or candidates

    kwargs = {"language": language}
    if gender:
        kwargs["gender"] = gender
    if candidates:
        kwargs["name"] = candidates[0]["name"]

    try:
        furhat.request_voice_config(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"voice config rejected ({exc}); continuing with {before}")
        return

    after = furhat.request_voice_status().get("voice_id")
    if verbose:
        if after == before and candidates:
            print(f"WARNING: voice still {after!r} after requesting {kwargs}. "
                  f"Run with --list-voices and pass --voice-name explicitly.")
        else:
            print(f"voice: {before!r} -> {after!r}")


def play(actions: list[dict], furhat, speed: float) -> None:
    t0 = time.monotonic()
    origin = actions[0]["t"] if actions else 0.0
    for action in actions:
        target_wall = (action["t"] - origin) / speed
        while (elapsed := time.monotonic() - t0) < target_wall:
            time.sleep(min(0.02, target_wall - elapsed))

        if action.get("location"):
            x, y, z = action["location"]
            furhat.request_attend_location(x, y, z)
        if action.get("gesture"):
            furhat.request_gesture_start(action["gesture"])
        if action.get("say"):
            # wait=False so speech does not block the next scheduled action and
            # drift the robot out of sync with the video playing alongside it
            furhat.request_speak_text(action["say"], wait=False)

        line = f"[{action['t']:7.2f}s] {action['kind']:8s} {action['target']} (score {action.get('score')})"
        if action.get("say"):
            line += f"  say: {action['say']!r}"
        print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="actions_05_15_s2.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--auth-key", default=None)
    parser.add_argument("--speed", type=float, default=1.0, help=">1 replays faster than real time")
    parser.add_argument("--language", default="en-GB", help="e.g. en-GB, en-US")
    parser.add_argument("--gender", default=None, help="male / female, if the SDK exposes it")
    parser.add_argument("--voice-name", default=None, help="exact voice name, overrides --language")
    parser.add_argument("--list-voices", action="store_true", help="print available voices and exit")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="pause before starting, so screen recording and video playback can be lined up")
    args = parser.parse_args()

    try:
        from furhat_realtime_api import FurhatClient
    except ImportError:
        raise SystemExit("pip install furhat_realtime_api")

    furhat = FurhatClient(args.host, auth_key=args.auth_key)
    furhat.set_logging_level(logging.WARNING)
    try:
        furhat.connect()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"could not connect to Furhat at {args.host}: {type(exc).__name__} {exc}\n"
            "If this timed out, the connection is probably not local: the Realtime API answers\n"
            "keyless auth only for localhost. Run this script on the SDK machine, or pass --auth-key."
        )

    if args.list_voices:
        status = furhat.request_voice_status()
        print(f"current: {status.get('voice_id')!r}")
        for v in _voice_entries(status):
            print(f"  {v['id']:30s} lang={v['language']:8s} gender={v['gender']}")
        furhat.disconnect()
        return

    if args.voice_name:
        furhat.request_voice_config(name=args.voice_name)
        print(f"voice set to {args.voice_name!r} -> {furhat.request_voice_status().get('voice_id')!r}")
    else:
        configure_voice(furhat, args.language, args.gender)

    actions = json.loads(Path(args.actions).read_text())
    print(f"{len(actions)} actions, {actions[0]['t']:.1f}s to {actions[-1]['t']:.1f}s")

    if args.countdown:
        for remaining in range(int(args.countdown), 0, -1):
            print(f"starting in {remaining}...", flush=True)
            time.sleep(1)

    play(actions, furhat, args.speed)
    furhat.disconnect()
    print("done")


if __name__ == "__main__":
    main()
