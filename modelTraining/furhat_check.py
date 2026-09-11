"""Interactive check of the three Furhat primitives the CAMEO demo relies on:
looking at a seat, performing a gesture, and speaking.

Run this on the SDK machine when the demo "did not work" but printed its whole
timeline -- that means the schedule was fine and one of the primitives below
silently did nothing. Each step announces itself, pauses, and asks you to watch
the robot, so the broken one is obvious.

Gesture names in particular cannot be listed through the API, and an unknown
name is accepted and ignored rather than raising, so the only way to find out
which exist in your SDK version is to try them and watch.

    py furhat_check.py                 # all three sections
    py furhat_check.py --only gestures
"""
from __future__ import annotations

import argparse
import logging
import time

# Standard Furhat built-ins. Any that this SDK does not have will simply do
# nothing -- that is the information we are after.
CANDIDATE_GESTURES = [
    "Nod", "Shake", "Smile", "BigSmile", "Blink", "Wink",
    "BrowRaise", "BrowFrown", "Surprise", "Thoughtful", "Oh", "GazeAway",
]

# The seat positions the demo uses, in the same order it uses them.
SEATS = [("A (left)", -0.5, 0.0, 1.2), ("B (centre)", 0.0, 0.0, 1.2), ("C (right)", 0.5, 0.0, 1.2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--auth-key", default=None)
    parser.add_argument("--voice-name", default="Amy")
    parser.add_argument("--only", choices=["attend", "gestures", "speech", "headpose"], default=None)
    parser.add_argument("--pause", type=float, default=2.5, help="seconds to hold each step")
    args = parser.parse_args()

    from furhat_realtime_api import FurhatClient

    furhat = FurhatClient(args.host, auth_key=args.auth_key)
    furhat.set_logging_level(logging.WARNING)
    furhat.connect()
    if args.voice_name:
        furhat.request_voice_config(name=args.voice_name)
    print(f"connected to {args.host}, voice = {furhat.request_voice_status().get('voice_id')!r}\n")

    def section(name: str) -> bool:
        return args.only is None or args.only == name

    if section("speech"):
        print("== SPEECH ==  you should hear one sentence in English")
        furhat.request_speak_text("Speech is working. This is the voice the demo will use.", wait=True)
        print("   done\n")

    if section("attend"):
        print("== ATTENTION ==  the head should turn LEFT, then CENTRE, then RIGHT")
        for label, x, y, z in SEATS:
            print(f"   attending {label}  ({x}, {y}, {z})")
            furhat.request_attend_location(x, y, z)
            time.sleep(args.pause)
        print("   done -- if the head never moved, attend_location is the problem\n")

    if section("headpose"):
        # Direct pose control bypasses the attention system entirely. If this
        # moves the head but attend_location did not, the issue is the
        # coordinate frame, not the connection.
        print("== HEAD POSE (direct) ==  head should turn left, right, then centre")
        for yaw in (-30.0, 30.0, 0.0):
            print(f"   yaw {yaw}")
            furhat.request_face_headpose(yaw, 0.0, 0.0, False)
            time.sleep(args.pause)
        print("   done\n")

    if section("gestures"):
        print("== GESTURES ==  watch the face; note which names actually do something")
        for name in CANDIDATE_GESTURES:
            print(f"   {name}")
            furhat.request_gesture_start(name)
            time.sleep(args.pause)
        print("   done -- names that produced no movement do not exist in this SDK\n")

    furhat.disconnect()
    print("finished")


if __name__ == "__main__":
    main()
