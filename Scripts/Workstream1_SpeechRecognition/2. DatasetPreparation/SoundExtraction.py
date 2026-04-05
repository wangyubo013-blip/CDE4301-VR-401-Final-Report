#!/usr/bin/env python3
"""
Extract an audio segment from a video into a WAV file.

Requirements:
- FFmpeg installed and available in PATH (ffmpeg, ffprobe).

Examples:
  python extract_audio_clip.py -i input.mp4 -s 00:01:12.500 -e 00:01:45.000 -o clip.wav
  python extract_audio_clip.py -i input.mkv -s 72.5 -e 105 -o clip.wav
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List


_TIME_RE = re.compile(r"^\d+(\.\d+)?$|^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")


def parse_time_to_seconds(t: str) -> float:
    """Accepts seconds ("12.3") or HH:MM:SS(.ms) ("01:02:03.400")."""
    t = t.strip()
    if not _TIME_RE.match(t):
        raise ValueError(f"Invalid time format: '{t}'. Use seconds (e.g., 12.5) or HH:MM:SS(.ms).")

    # numeric seconds
    if ":" not in t:
        return float(t)

    # HH:MM:SS(.ms)
    hh, mm, ss = t.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def seconds_to_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    # Keep milliseconds if present
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def run_cmd(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and ensure 'ffmpeg' is in your PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nFFmpeg error:\n{err}") from e


def main():
    p = argparse.ArgumentParser(description="Extract an audio segment from a video into WAV.")
    p.add_argument("-i", "--input", required=True, help="Input video file path.")
    p.add_argument("-s", "--start", required=True, help="Start time (seconds or HH:MM:SS[.ms]).")
    p.add_argument("-e", "--end", required=True, help="End time (seconds or HH:MM:SS[.ms]).")
    p.add_argument("-o", "--output", required=True, help="Output WAV file path.")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz). Default: 16000")
    p.add_argument("--mono", action="store_true", help="Force mono output.")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    start_sec = parse_time_to_seconds(args.start)
    end_sec = parse_time_to_seconds(args.end)

    if end_sec <= start_sec:
        print("End time must be greater than start time.", file=sys.stderr)
        sys.exit(1)

    duration_sec = end_sec - start_sec

    # Build FFmpeg command:
    # -ss before -i for faster seek (good enough for most cases)
    # -t duration to limit segment
    # output WAV PCM 16-bit
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", seconds_to_hhmmss(start_sec),
        "-i", str(in_path),
        "-t", f"{duration_sec:.6f}",
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(args.sr),
    ]

    if args.mono:
        cmd += ["-ac", "1"]

    cmd += [str(out_path)]

    run_cmd(cmd)

    print("=== Extraction complete ===")
    print(f"Output WAV     : {out_path.resolve()}")
    print(f"Start time     : {seconds_to_hhmmss(start_sec)}  ({start_sec:.3f} s)")
    print(f"End time       : {seconds_to_hhmmss(end_sec)}  ({end_sec:.3f} s)")
    print(f"Duration       : {seconds_to_hhmmss(duration_sec)}  ({duration_sec:.3f} s)")


if __name__ == "__main__":
    main()
