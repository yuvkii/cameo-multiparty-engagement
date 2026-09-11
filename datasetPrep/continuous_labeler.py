"""Continuous, whole-session engagement labeling via a live-recorded slider.

Unlike manual_labeler.py (which scores one 3s segment/person at a time from
a fixed set of buckets), this plays an ENTIRE session's Cam_2 footage back
in real time while the reviewer continuously drags a small on-screen slider
to track one participant's engagement as it rises and falls -- the slider's
position is sampled every displayed frame and written out as a full,
frame-by-frame trace. Rationale: quick within-segment attention shifts are
easy to miss when forced into one score per 3s window; watching continuously
and "recording" a moving judgment (the same idea behind continuous affect
annotation tools like FEELTRACE/GTrace) captures the shape of an engagement
change, not just its endpoints.

Run once per participant per session (3 passes for 3 real participants --
the robot confederate is not scored). The reviewer picks ONE physical person
to visually follow for the whole pass and ignores everyone else in frame --
this sidesteps the track_id/person_idx drift problem this project has hit
repeatedly, since a human doesn't lose track of who's who just because a
bounding box briefly breaks, unlike the automated re-identification used
elsewhere in this codebase.

NOTE -- open problem, not solved by this script: reconciling a continuous
per-participant trace back to a specific detected entity (person_idx/bbox)
frame-by-frame, the way manual_labeler.py's position-ranking does per
segment, still needs a separate later step once whole-session tracking is
attempted. Full-session automated re-identification is untested at this
duration and may fragment more than the ~2.3%-of-segments rate already
measured within single 3s windows -- don't assume it "just works" without
checking.

Usage:
    dataExtraction/openface3/.venv/bin/python3 \
        datasetPrep/continuous_labeler.py --dataset 05_15 --session 1 --participant A

Controls:
    drag the bar (bottom-center) with the mouse, or press a/d to nudge
        the value down/up a couple of points at a time -- sets the CURRENT
        engagement % for whichever one participant you're following. The
        big percentage readout sits right above the bar, in the same
        bottom-center cluster as the frame/time status line, so you don't
        have to look away to a corner to see what value you're giving.
    k = speed up playback, j = slow it down (live, no need to restart with
        a different --speed) -- current multiplier shown in the status line
    space = pause/resume playback (slider stays interactive while paused,
        but nothing is recorded until you resume)
    r = rewind 5s and re-record over that stretch (discards previously
        recorded samples from that point forward)
    q = quit and save (resumable -- re-run the same command to continue
        from the last recorded frame)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import pandas as pd

from render_review_clips import open_cam2_capture

PROJECT_ROOT = Path(__file__).parent.parent
WINDOW = "continuous_labeler"

NUDGE_PCT = 2.0
SPEED_MIN, SPEED_MAX = 0.1, 8.0  # 8x is about the real ceiling; see SEEK_CROSSOVER_FRAMES
SPEED_STEP_UP, SPEED_STEP_DOWN = 1.25, 0.8  # inverse of each other, so k then j round-trips back

# Advancing the video is the loop's real cost, and the two ways of doing it
# scale differently (both measured on 05_14's 2048x1088 footage):
#   grab() per frame      ~3.2ms EACH  -- linear in how far you jump
#   set(POS_FRAMES) seek  ~50ms FLAT   -- same cost for 2 frames or 200
# So chain grab()s for short hops and seek for long ones. They cross over
# around 16 frames; past that, seeking is what makes high --speed values
# actually reachable instead of the loop just falling behind.
SEEK_CROSSOVER_FRAMES = 16
# Displayed frames per second to aim for. Playback speed is achieved by
# showing FEWER, further-apart frames rather than by trying to draw every
# frame faster -- drawing is what the loop can't do quickly enough. Drops at
# high speed because ~18Hz of very large jumps is affordable where 30Hz is
# not, and while scrubbing fast a choppier picture is an acceptable trade.
TARGET_DISPLAY_FPS_NORMAL = 30.0
TARGET_DISPLAY_FPS_FAST = 18.0


def plan_advance(fps: float, speed: float) -> tuple[int, bool]:
    """How many frames to move per displayed frame, and whether to get there
    by seeking rather than by chaining grab()s."""
    target_display = TARGET_DISPLAY_FPS_NORMAL if speed <= 2.0 else TARGET_DISPLAY_FPS_FAST
    stride = max(1, int(round(fps * speed / target_display)))
    return stride, stride > SEEK_CROSSOVER_FRAMES


class Slider:
    """Bottom-center bar + large percentage readout, positioned relative to
    the actual frame size (varies by dataset -- 2048x1088 vs 1280x1024) so it
    always sits in the same bottom cluster as the status line below it,
    rather than a fixed pixel offset that could land wrong on a smaller
    frame. Moved here from the top-left corner (2026-07-28, user feedback:
    the corner was too far from where a followed participant's face
    typically is to glance at while tracking them)."""

    def __init__(self, frame_w: int, frame_h: int, initial_pct: float = 50.0):
        self.bar_w, self.bar_h = 300, 24
        self.bar_x = (frame_w - self.bar_w) // 2
        self.bar_y = frame_h - 90
        self.pct = initial_pct
        self._dragging = False

    def handle_mouse(self, event, x, y, flags, _param) -> None:
        inside = (self.bar_x <= x <= self.bar_x + self.bar_w
                  and self.bar_y - 10 <= y <= self.bar_y + self.bar_h + 10)
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self._dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        if self._dragging and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            frac = (x - self.bar_x) / self.bar_w
            self.pct = float(min(100.0, max(0.0, round(frac * 100))))

    def nudge(self, delta: float) -> None:
        self.pct = float(min(100.0, max(0.0, self.pct + delta)))

    def draw(self, frame) -> None:
        x0, y0, w, h = self.bar_x, self.bar_y, self.bar_w, self.bar_h
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (50, 50, 50), -1)
        fill_w = int(w * self.pct / 100)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + h), (0, 200, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 2)

        label = f"{self.pct:.0f}%"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
        (tw, _th), _ = cv2.getTextSize(label, font, scale, thick)
        tx, ty = x0 + w // 2 - tw // 2, y0 - 16
        cv2.putText(frame, label, (tx, ty), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), font, scale, (0, 255, 0), thick, cv2.LINE_AA)


def _bottom_overlay(frame, text: str) -> None:
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. 05_15")
    parser.add_argument("--session", required=True, type=int)
    parser.add_argument("--participant", required=True, choices=["A", "B", "C"],
                         help="which real participant this pass follows -- pick one physical person and track only them")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="playback speed multiplier, e.g. 0.5 = half speed if you need more reaction time")
    args = parser.parse_args()

    # Multi-part aware: a long take split by AVI's ~4GB limit (05_14) plays
    # as one continuous stream with session-wide frame numbering, so the
    # trace's frame_index still joins to the other pipelines. Opening only
    # the first part would silently label a fraction of the session.
    cap = open_cam2_capture(args.dataset, args.session)
    if cap is None:
        raise SystemExit(f"no Cam_2 footage found for {args.dataset} session {args.session}")
    if len(cap.paths) > 1:
        parts = ", ".join(f"{p.name} ({n} frames)" for p, n in zip(cap.paths, cap.counts))
        print(f"session {args.session} is split across {len(cap.paths)} files, playing as one: {parts}")

    out_dir = PROJECT_ROOT / "outputs" / args.dataset / "continuous_labels" / f"session_{args.session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"participant_{args.participant}.csv"

    if not cap.isOpened():
        raise SystemExit(f"could not open {cap.paths[0]}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # UI poll rate is decoupled from playback speed -- redraw/read-mouse at a
    # fixed ~66Hz regardless of speed, and only advance the actual video
    # frame when enough wall-clock time has passed for the requested speed.
    # Coupling them (as an earlier version did) meant slow speeds halved the
    # slider's redraw rate too, which is what made it feel laggy.
    speed = args.speed
    # The loop cannot draw every frame fast enough at high fps or high speed
    # (one iteration costs ~21ms: waitKey's 15ms floor + ~5ms decode + draw,
    # while 60fps allows only 16.7ms per frame -- which is why 05_14 played
    # below 1x before this). Playback rate is therefore achieved by showing
    # fewer, further-apart frames, not by drawing faster. A value is still
    # RECORDED for every frame index (see the advance block below), so the
    # trace stays frame-for-frame complete and joins to the other pipelines
    # -- engagement cannot meaningfully change within a skipped span.
    display_stride, use_seek = plan_advance(fps, speed)
    frame_period_sec = display_stride / (fps * speed)
    # waitKey's floor is a fixed tax per displayed frame; at speed it's a
    # large slice of the budget, and fine slider control matters less while
    # scrubbing than keeping up does.
    def poll_ms_for(sp: float) -> int:
        return 15 if sp <= 2.0 else 1
    POLL_MS = poll_ms_for(speed)

    rows: list[dict] = []
    start_frame = 0
    initial_pct = 50.0
    if out_path.exists():
        prior = pd.read_csv(out_path)
        rows = prior.to_dict("records")
        if rows:
            start_frame = int(rows[-1]["frame_index"]) + 1
            initial_pct = float(rows[-1]["engagement_pct"])
        print(f"Resuming {out_path.name} from frame {start_frame}/{total_frames}")

    if start_frame >= total_frames:
        print("Already fully labeled for this participant/session.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    slider = Slider(frame_w, frame_h, initial_pct)
    cv2.setMouseCallback(WINDOW, slider.handle_mouse)

    print(f"session {args.session}, participant {args.participant}: "
          f"{total_frames - start_frame} frames left ({(total_frames - start_frame) / fps:.0f}s of footage).")
    print("Drag the bottom-center bar or press a/d to adjust.  k/j = speed up/down  "
          "space=pause  r=rewind 5s  q=quit&save")

    paused = False
    frame_idx = start_frame
    frame = None
    last_drawn = None
    save_every = max(1, int(fps * 5))
    next_frame_due = time.monotonic()

    while frame_idx < total_frames:
        advanced = 0
        if not paused and time.monotonic() >= next_frame_due:
            if use_seek:
                # One flat-cost jump beats hundreds of grab()s at this range.
                target = min(frame_idx + display_stride - 1, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, next_frame = cap.read()
                if ret:
                    frame = next_frame
                    advanced = target - frame_idx + 1
                else:
                    break
            else:
                # Step over the frames we won't draw (cheap, no decode),
                # then decode only the one actually shown.
                for _ in range(display_stride - 1):
                    if not cap.grab():
                        break
                    advanced += 1
                ret, next_frame = cap.read()
                if ret:
                    frame = next_frame
                    advanced += 1
                elif advanced == 0:
                    break
            next_frame_due += frame_period_sec
            if next_frame_due < time.monotonic():
                # Fell behind (advance cost more than the period). Reset to
                # NOW, not now+period: the latter idles a further full period
                # on top of an advance that was already too slow, which
                # roughly halved throughput exactly when the loop was
                # saturated. "now" means run flat out while saturated, and
                # still never replays a burst of skipped frames to catch up.
                next_frame_due = time.monotonic()

        # Only repaint when something actually changed. The copy alone is a
        # 6.7MB memcpy on 2048x1088 footage, and at high speed POLL_MS drops
        # to 1ms, so repainting unconditionally burned more time between
        # frame advances than the advance itself -- which is what stopped
        # high --speed values from being reachable.
        draw_state = (frame_idx, slider.pct, paused, round(speed, 3))
        if draw_state != last_drawn:
            display = frame.copy()
            slider.draw(display)
            _bottom_overlay(
                display,
                f"session {args.session}  participant {args.participant}  "
                f"frame {frame_idx}/{total_frames}  ({frame_idx / fps:.1f}s/{total_frames / fps:.1f}s)  "
                f"speed={speed:.2f}x"
                + (f"  [{fps:.0f}fps, showing 1/{display_stride}]" if display_stride > 1 else "")
                + ("  [PAUSED]" if paused else ""),
            )
            cv2.imshow(WINDOW, display)
            last_drawn = draw_state
        key = cv2.waitKey(POLL_MS) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            next_frame_due = time.monotonic() + frame_period_sec
            continue
        if key == ord("a"):
            slider.nudge(-NUDGE_PCT)
        elif key == ord("d"):
            slider.nudge(NUDGE_PCT)
        elif key in (ord("k"), ord("j")):
            speed = (min(SPEED_MAX, speed * SPEED_STEP_UP) if key == ord("k")
                     else max(SPEED_MIN, speed * SPEED_STEP_DOWN))
            # Stride and advance method are functions of speed, so they have
            # to be replanned here -- only rescaling the period would leave
            # the loop asking for more frames than it can deliver.
            display_stride, use_seek = plan_advance(fps, speed)
            frame_period_sec = display_stride / (fps * speed)
            POLL_MS = poll_ms_for(speed)
        elif key == ord("r"):
            rewind_frames = int(5 * fps)
            frame_idx = max(0, frame_idx - rewind_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            rows = [r for r in rows if r["frame_index"] < frame_idx]
            next_frame_due = time.monotonic()
            continue

        if not paused and advanced:
            # One row per real frame even when several were stepped over
            # for display -- they all carry the slider value in force while
            # they were on screen.
            for _ in range(advanced):
                rows.append({"frame_index": frame_idx, "timestamp_sec": frame_idx / fps,
                             "engagement_pct": slider.pct})
                frame_idx += 1
                if frame_idx % save_every == 0:
                    pd.DataFrame(rows).to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    pd.DataFrame(rows).to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    cap.release()
    cv2.destroyAllWindows()
    remaining = total_frames - frame_idx
    print(f"Saved {len(rows)} frame-level samples to {out_path}.")
    if remaining > 0:
        print(f"{remaining} frame(s) left -- re-run the same command to resume.")
    else:
        print("Pass complete for this participant.")


if __name__ == "__main__":
    main()
