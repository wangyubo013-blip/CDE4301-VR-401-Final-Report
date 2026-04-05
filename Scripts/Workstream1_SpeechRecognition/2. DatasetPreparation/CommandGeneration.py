#!/usr/bin/env python3
"""
CommandGeneration.py

Reads a pure text CSV timeline file and fills column 6 (F) for rows 2..N
with the SoundExtraction command.

CSV format (no header):

Row 1 (at least 5 entries):
  col1 = source file name (under Sources/)
  col2 = folder name under Extraction/
  col3 = clip prefix
  col4 = "Real"   (label/value; not used for command generation)
  col5 = "Azure"  (label/value; not used for command generation)
  col6.. = optional extra fields

Row 2..N (at least 5 columns):
  col1 = clip numbering (integer)
  col2 = clip start time in mm:ss
  col3 = clip end time in mm:ss
  col4 = correct reading (free text)
  col5 = another sentence (already exists)
  col6 = (will be filled/overwritten with the generated command)

Usage:
  python CommandGeneration.py -i Timeline.csv -o Timeline_filled.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def mmss_to_seconds(t: str) -> int:
    """Convert 'mm:ss' string to integer seconds."""
    s = str(t).strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Time must be mm:ss, got: {t!r}")
    mm, ss = parts
    return int(mm) * 60 + int(ss)


def generate_commands(rows, python_exe, script_name, sources_dir, extraction_root):
    if not rows or len(rows[0]) < 5:
        raise ValueError(
            "Row 1 must have at least 5 entries: source file, extraction folder, clip prefix, Real, Azure."
        )

    source_file = rows[0][0].strip()
    folder_name = rows[0][1].strip()
    clip_prefix = rows[0][2].strip()
    # rows[0][3] = Real (not used)
    # rows[0][4] = Azure (not used)

    for i in range(1, len(rows)):
        row = rows[i]

        # Need at least 5 columns because col5 exists and col6 will be written
        if len(row) < 5:
            raise ValueError(
                f"Row {i+1} must have at least 5 columns: clip_no,start,end,correct_reading,<existing col5>"
            )

        clip_no_raw = str(row[0]).strip()
        start_raw = str(row[1]).strip()
        end_raw = str(row[2]).strip()

        try:
            clip_no = int(clip_no_raw)
        except ValueError as e:
            raise ValueError(f"Row {i+1}: clip numbering must be an integer, got {clip_no_raw!r}") from e

        start_sec = mmss_to_seconds(start_raw)
        end_sec = mmss_to_seconds(end_raw)

        if end_sec <= start_sec:
            raise ValueError(f"Row {i+1}: end time must be greater than start time ({start_raw} -> {end_raw}).")

        clip_no_padded = f"{clip_no:02d}"

        in_path = f'{sources_dir}/{source_file}'
        out_path = f'{extraction_root}/{folder_name}/{clip_prefix}-{clip_no_padded}.wav'

        cmd = (
            f'{python_exe} {script_name} '
            f'-i "{in_path}" -s {start_sec} -e {end_sec} '
            f'-o "{out_path}"'
        )

        # Ensure column 6 exists
        while len(row) < 6:
            row.append("")
        row[5] = cmd

    return rows


def main():
    p = argparse.ArgumentParser(description="Fill column 6 of a timeline CSV with SoundExtraction commands.")
    p.add_argument("-i", "--input", required=True, help="Input timeline CSV (pure CSV text).")
    p.add_argument("-o", "--output", required=True, help="Output CSV with commands filled in.")
    p.add_argument("--python", dest="python_exe", default="python", help='Python executable string. Default: "python"')
    p.add_argument("--script", dest="script_name", default="SoundExtraction.py", help='Extraction script name. Default: "SoundExtraction.py"')
    p.add_argument("--sources", dest="sources_dir", default="Sources", help='Sources directory. Default: "Sources"')
    p.add_argument("--extraction", dest="extraction_root", default="Extraction", help='Extraction root directory. Default: "Extraction"')
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    with in_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    try:
        rows = generate_commands(
            rows,
            python_exe=args.python_exe,
            script_name=args.script_name,
            sources_dir=args.sources_dir,
            extraction_root=args.extraction_root,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print("=== Command generation complete ===")
    print(f"Input  : {in_path.resolve()}")
    print(f"Output : {out_path.resolve()}")
    print(f"Filled : {max(0, len(rows)-1)} command rows")
    print("Wrote commands to column 6 (F).")


if __name__ == "__main__":
    main()
