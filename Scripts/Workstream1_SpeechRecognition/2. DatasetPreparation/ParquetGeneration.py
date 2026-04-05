from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wave
import sys

# =========================
# READ ME
# Command: Python ParquetGeneration.py xx
# Replace x with the index of the sample to generate the corresponding Parquet dataset.
# =========================


# =========================
# Config
# =========================
SAMPLE_PREFIX= sys.argv[1] if len(sys.argv) > 1 else "00" #edit

PROJECT_DIR = Path(__file__).parent
SAMPLE_DIR = PROJECT_DIR / "Extraction" / f"Sample{SAMPLE_PREFIX}"
CSV_PATH = SAMPLE_DIR / "Timeline.csv"
OUTPUT_FILE="Sample"+SAMPLE_PREFIX+".parquet"
OUTPUT_PATH = SAMPLE_DIR / OUTPUT_FILE
AUDIO_PREFIX = "clip"+SAMPLE_PREFIX      # clip01-01.wav, clip01-02.wav, ...
AUDIO_EXT = ".wav"
AUDIO_DIGITS = 2             # 01, 02, ... 16

# Which transcript column to use from CSV: Real
# - if your CSV has "Real" / "Azure", pick one
# - otherwise use "text" or "Transcript"
PREFERRED_TEXT_COLUMNS = ["Real"]

# Optional timestamp columns (set to None if not present)
START_COL_CANDIDATES = ["segment_start_time", "start", "Start", "start_time", "StartTime"]
END_COL_CANDIDATES   = ["segment_end_time", "end", "End", "end_time", "EndTime"]
DUR_COL_CANDIDATES   = ["duration", "Duration", "dur", "Dur"]

# Parquet writing parameters
BATCH_SIZE = 512                 # reduce if you have huge audio files
COMPRESSION = "zstd"             # "zstd" (best) or "snappy" (fast)
COMPRESSION_LEVEL = 6            # only used for zstd/gzip
# =========================


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def wav_duration_seconds(wav_path: Path) -> float:
    # Reads header only (fast, no full decode)
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate == 0:
        return float("nan")
    return frames / rate


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Timeline.csv not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    text_col = pick_first_existing_column(df, PREFERRED_TEXT_COLUMNS)
    if text_col is None:
        raise ValueError(
            f"No transcript column found. Tried: {PREFERRED_TEXT_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    start_col = pick_first_existing_column(df, START_COL_CANDIDATES)
    end_col = pick_first_existing_column(df, END_COL_CANDIDATES)
    dur_col = pick_first_existing_column(df, DUR_COL_CANDIDATES)

    # Define Arrow schema to match HF-style
    schema = pa.schema([
        ("id", pa.string()),
        ("audio", pa.struct([
            ("bytes", pa.binary()),
            ("path", pa.string()),
        ])),
        ("text", pa.string()),
        ("segment_start_time", pa.float32()),
        ("segment_end_time", pa.float32()),
        ("duration", pa.float32()),
    ])

    # Streaming Parquet writer (batching avoids high RAM)
    writer = pq.ParquetWriter(
        where=str(OUTPUT_PATH),
        schema=schema,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL if COMPRESSION in ("zstd", "gzip") else None,
        use_dictionary=True
    )

    try:
        n = len(df)
        for batch_start in range(0, n, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n)

            ids = []
            audio_bytes_list = []
            audio_paths = []
            texts = []
            starts = []
            ends = []
            durs = []

            for idx in range(batch_start, batch_end):
                clip_index = idx + 1
                audio_name = f"{AUDIO_PREFIX}-{clip_index:0{AUDIO_DIGITS}d}{AUDIO_EXT}"
                audio_path = SAMPLE_DIR / audio_name
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing audio file: {audio_name}")

                # Embed raw wav bytes (no decoding)
                # b = audio_path.read_bytes()

                # Embed raw wav bytes (decoding to mono)
                import soundfile as sf
                import io

                data, sr = sf.read(audio_path)

                # If stereo → average channels
                if len(data.shape) == 2:
                    data = data.mean(axis=1)
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                buf = io.BytesIO()
                sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
                b = buf.getvalue()

                # Text
                t = str(df.iloc[idx][text_col]).strip()

                # Times/duration
                st = float(df.iloc[idx][start_col]) if start_col else float("nan")
                et = float(df.iloc[idx][end_col]) if end_col else float("nan")

                if dur_col:
                    du = float(df.iloc[idx][dur_col])
                else:
                    du = wav_duration_seconds(audio_path)

                ids.append(f"{AUDIO_PREFIX}_{clip_index:04d}")
                audio_bytes_list.append(b)
                audio_paths.append(audio_name)
                texts.append(t)
                starts.append(st)
                ends.append(et)
                durs.append(du)

            batch_table = pa.Table.from_arrays(
                [
                    pa.array(ids, type=pa.string()),
                    pa.StructArray.from_arrays(
                        [
                            pa.array(audio_bytes_list, type=pa.binary()),
                            pa.array(audio_paths, type=pa.string()),
                        ],
                        fields=schema.field("audio").type
                    ),
                    pa.array(texts, type=pa.string()),
                    pa.array(starts, type=pa.float32()),
                    pa.array(ends, type=pa.float32()),
                    pa.array(durs, type=pa.float32()),
                ],
                schema=schema
            )

            writer.write_table(batch_table)

            print(f"Wrote rows {batch_start}..{batch_end-1} ({batch_end}/{n})")

    finally:
        writer.close()

    print(f"\n✓ Done. Parquet written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
