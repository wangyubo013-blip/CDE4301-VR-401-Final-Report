#from datasets import load_dataset
#ds = load_dataset("Jzuluaga/atcosim_corpus", split="train")
#print(next(iter(ds))["text"][:120])
# from datasets import load_dataset
#
# ds = load_dataset(
#     "Jzuluaga/atcosim_corpus",
#     split="train",
#     download_mode="reuse_dataset_if_exists",   # important
# )
#
#
# print(ds)
# print(ds[0].keys())
# print(ds[0])

# from datasets import load_dataset, Audio
# from transformers import WhisperProcessor
#
# model_name = "openai/whisper-large-v3"
# processor = WhisperProcessor.from_pretrained(model_name)
#
# ds = load_dataset("Jzuluaga/atcosim_corpus", split="train")
# ds = ds.cast_column("audio", Audio(sampling_rate=16000))
#
# def prepare(batch):
#     audio = batch["audio"]
#     inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])
#     batch["input_features"] = inputs["input_features"][0]
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#     return batch
#
# ds_prep = ds.map(prepare, remove_columns=ds.column_names, num_proc=4)

import io
import soundfile as sf
from datasets import load_dataset, Audio
import numpy as np
from transformers import WhisperProcessor

PARQUET_PATH = "/root/autodl-tmp/whisper_project/whisper_training/Sample03.parquet"

# Load parquet
dataset = load_dataset(
    "parquet",
    data_files={"train": PARQUET_PATH},
)

train_ds = dataset["train"]

print(train_ds)
print(train_ds[0].keys())
print(train_ds[0]["audio"].keys())


def decode_audio_from_bytes(example):
    audio_bytes = example["audio"]["bytes"]

    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

    # Convert stereo to mono if needed
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    example["audio_decoded"] = {
        "array": audio_array,
        "sampling_rate": sampling_rate,
    }
    return example

train_ds = train_ds.map(decode_audio_from_bytes)




BASE_MODEL = "/root/autodl-tmp/whisper_project/whisper_training/models/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(BASE_MODEL, local_files_only=True)

def prepare_dataset(example):
    audio_array = np.array(example["audio_decoded"]["array"], dtype=np.float32)
    sampling_rate = example["audio_decoded"]["sampling_rate"]

    example["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate
    ).input_features[0]

    example["labels"] = processor.tokenizer(
        example["text"],
        truncation=True
    ).input_ids

    return example

processed_ds = train_ds.map(
    prepare_dataset,
    remove_columns=train_ds.column_names,
    num_proc=1
)

print(processed_ds)
print(processed_ds[0].keys())
print(type(processed_ds[0]["input_features"]))
print(len(processed_ds[0]["input_features"]))      # should be 128 for large-v3
print(len(processed_ds[0]["input_features"][0]))   # usually 3000
print(processed_ds[0]["labels"][:20])

processed_ds.save_to_disk(
    "/root/autodl-tmp/whisper_project/whisper_training/processed_dataset-03"
)