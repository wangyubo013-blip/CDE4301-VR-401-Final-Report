import pyarrow.parquet as pq
from datasets import load_dataset
import soundfile as sf
import io
from pathlib import Path


parquet_file = "Sample03.parquet"
out_dir = Path("check")
out_dir.mkdir(exist_ok=True)

pf = pq.ParquetFile(parquet_file)

print("Schema:")
print(pf.schema)

print("\nNumber of rows:", pf.metadata.num_rows)
print("Number of row groups:", pf.num_row_groups)

table = pq.read_table(
    parquet_file,
    columns=["text"]   # IMPORTANT: select columns
)

df = table.to_pandas()
print(df.head(5))

table = pq.read_table(parquet_file)
print(table.schema)

#full loading
ds = load_dataset(
    "parquet",
    data_files=parquet_file,
    split="train"
)

for i in range(pf.metadata.num_rows):
    sample = ds[i]
    print(f"\n===== Record {i} =====")
    print("id:", sample["id"])
    print("text:", sample["text"])
    print("segment_start_time:", sample["segment_start_time"])
    print("segment_end_time:", sample["segment_end_time"])
    print("duration:", sample["duration"])

    audio = sample["audio"]
    print("audio.path:", audio.get("path"))
    print(
        "audio.bytes:",
        f"{len(audio['bytes'])} bytes" if audio.get("bytes") is not None else None
    )

    wav_path = out_dir / f"sample_{i}.wav"
    with sf.SoundFile(io.BytesIO(audio["bytes"])) as f:
        data = f.read(dtype="float32")
        sr = f.samplerate
        sf.write(wav_path, data, sr)

    print(f"Saved: {wav_path}")
    print("Text:", ds[i]["text"])