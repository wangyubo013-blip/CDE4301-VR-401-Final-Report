from faster_whisper import WhisperModel
import numpy as np
import re
import pandas as pd
from datasets import Dataset, Audio
import evaluate
import tempfile
import soundfile as sf
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

from text_to_num import alpha2digit

def collapse_spaced_digits(text: str) -> str:
    # collapse sequences like "6 1 6" -> "616"
    return re.sub(r"\b\d(?:\s+\d)+\b", lambda m: m.group(0).replace(" ", ""), text)

def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = alpha2digit(text, "en")
    text = collapse_spaced_digits(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
TEST_PARQUET = "/root/autodl-tmp/whisper_project/whisper_training/CombinedSamples_01_03_04_05_06_07.parquet"

# =========================
# 1) Init model
# =========================
model = WhisperModel(
    "/root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-ct2-continued-3",
    #"/root/autodl-tmp/whisper_project/faster-whisper/faster-whisper-large-v3",
    device="cuda",
    compute_type="float16"
)

# Force mel filterbank to match model.n_mels
model.feature_extractor.mel_filters = model.feature_extractor.get_mel_filters(
    model.feature_extractor.sampling_rate,
    model.feature_extractor.n_fft,
    n_mels=model.model.n_mels
).astype(np.float32)

print("model n_mels:", model.model.n_mels)
print("mel_filters shape:", model.feature_extractor.mel_filters.shape)

# 2) Init evaluate model
#wer_metric = evaluate.load("wer")
wer_metric = evaluate.load("/root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--wer/e41eaa77ca7152430cd94704de20946c1b004b5b488ab5d20b26fb81c6c15506/wer.py")

# =========================
# 3) Load test set
# =========================
ds = Dataset.from_parquet(TEST_PARQUET)
ds = ds.cast_column("audio", Audio())

print(ds)
print(ds.features)
print(ds[0])

# =========================
# 4) text normalization
# =========================
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", "", s)   # remove punctuation, keep apostrophe
    s = re.sub(r"\s+", " ", s)
    return s

# Change these to match your parquet
AUDIO_COL = "audio"
TEXT_COL = "text"

'''
# for only 1 audio test
segments, info = model.transcribe("test.wav", language="en")
for segment in segments:
    print(segment.text)
'''

# =========================
# 5) Transcribe + collect refs/preds
# =========================
references = []
predictions = []
rows = []

initial_prompt = (
    "cockpit conversation between a pilot and copilot during flight"
    "runway, downwind, tower, knots, altitude, gear, speedbrake, checklist, vs, fds, Singapore, alt hold, heading sel"
)

for i, sample in enumerate(ds):
    ref_text = sample["text"]
    audio_info = sample["audio"]

    # audio_info should contain array + sampling_rate after cast_column(Audio())
    audio_array = audio_info["array"]
    sr = audio_info["sampling_rate"]

    # faster-whisper transcribe() usually wants a file path,
    # so save temp wav for each sample
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio_array, sr)

        segments, info = model.transcribe(
            f.name,
            language="en",
            task="transcribe",
            beam_size=1,
            initial_prompt = initial_prompt,

        )

        pred_text = " ".join(seg.text for seg in segments).strip()

    ref_norm = normalize_for_wer(normalizer(ref_text))
    pred_norm = normalize_for_wer(normalizer(pred_text))

    references.append(ref_norm)
    predictions.append(pred_norm)

    rows.append({
        "id": sample.get("id", i),
        "reference": ref_text,
        "prediction": pred_text,
        "reference_norm": ref_norm,
        "prediction_norm": pred_norm,
        "duration": sample.get("duration", None),
        "segment_start_time": sample.get("segment_start_time", None),
        "segment_end_time": sample.get("segment_end_time", None),
    })

    print(f"[{i}] REF : {ref_norm}")
    print(f"[{i}] PRED: {pred_norm}")
    print("-" * 60)

# =========================
# 6) Compute WER
# =========================
#norm_preds = [normalizer(p) for p in predictions]
#norm_refs  = [normalizer(r) for r in references]

#wer = wer_metric.compute(predictions=norm_preds, references=norm_refs)
wer = wer_metric.compute(predictions=predictions, references=references)
print(f"\nWER = {wer:.4f}")
print(f"WER% = {wer * 100:.2f}%")

# =========================
# 7) Save detailed results
# =========================
out_df = pd.DataFrame({
    "reference": references,
    "prediction": predictions
})
out_df["match"] = out_df["reference"] == out_df["prediction"].to_csv("wer_LoRA_w_prompt.csv", index=False)
#print("Saved detailed results to wer_results.csv")