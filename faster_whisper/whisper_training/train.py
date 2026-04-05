import os
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

MODEL_ID = "/root/autodl-tmp/whisper_project/whisper_training/models/whisper-large-v3"
CKPT_DIR = "./whisper-large-v3-atcosim-lora/checkpoint-700"
OUT_DIR = "./whisper-large-v3-atcosim-lora"

#################################
# 重要！！！！！ To avoid Internet Connection Problem:
# Run the following in Terminal before "python train.py" if dataset and model are already cached
'''
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
'''
#####################################

# 1) Load dataset (already cached after your split generation)
print("Step 1")
train = load_dataset("Jzuluaga/atcosim_corpus", split="train",download_mode="reuse_dataset_if_exists")
test  = load_dataset("Jzuluaga/atcosim_corpus", split="test",download_mode="reuse_dataset_if_exists")

train = train.cast_column("audio", Audio(sampling_rate=16000))
test  = test.cast_column("audio",  Audio(sampling_rate=16000))

# 2) Load processor + base model
print("Step 2")
MODEL_PATH = "/root/autodl-tmp/whisper_project/whisper_training/models/whisper-large-v3"

processor = WhisperProcessor.from_pretrained(MODEL_PATH,local_files_only=True) #use cached version is faster?
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH,use_safetensors=True,local_files_only=True)
model = PeftModel.from_pretrained(
    base_model,
    CKPT_DIR,
    is_trainable=True,
)

# After: model = WhisperForConditionalGeneration.from_pretrained(...)
gen_cfg = model.generation_config

# Don't suppress any tokens
gen_cfg.suppress_tokens = None          # or [] depending on your preference; None is safest in v5

# Don't force any decoder ids (language/task tokens)
gen_cfg.forced_decoder_ids = None

# Optional: since you're doing English transcription
gen_cfg.language = "en"
gen_cfg.task = "transcribe"

model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

model.generation_config = gen_cfg

# 3) Attach LoRA adapter (ONLY these params will train)
print("Step 3")
'''
#only needed for new training
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
'''
model.print_trainable_parameters()

# 4) Preprocess: audio -> input_features, text -> labels
print("Step 4")
def preprocess(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

remove_cols = train.column_names  # ['id','audio','text','segment_start_time','segment_end_time','duration']
train_p = train.map(preprocess, remove_columns=remove_cols, num_proc=1)
test_p  = test.map(preprocess,  remove_columns=remove_cols, num_proc=1)

# 5) Data collator (pads features + labels)
print("Step 5")
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        batch["input_features"] = batch["input_features"].half()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 6) WER metric
print("Step 6")
#wer_metric = evaluate.load("wer")
#if cached and want to avoid connect to Hugging Face
wer_metric = evaluate.load("/root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--wer/e41eaa77ca7152430cd94704de20946c1b004b5b488ab5d20b26fb81c6c15506/wer.py")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # Ensure pad token id is defined
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# 7) Training args (safe defaults for large-v3 LoRA)
print("Step 7")
# If you get CUDA OOM: set per_device_train_batch_size=1 and increase grad_accum_steps.
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=2000,
    fp16=True,

    eval_strategy="steps",   # <-- change THIS
    eval_steps=200,
    save_steps=100,
    logging_steps=25,

    predict_with_generate=True,
    generation_max_length=225,
    report_to="none",
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_p,
    eval_dataset=test_p,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()
#PyTorch 2.5.1 did not support resume_from_checkpoint
#Solution: Rebuild trainer from the checkpoint
#trainer.train(resume_from_checkpoint="./whisper-large-v3-atcosim-lora/checkpoint-700")

# 8) Save ONLY the LoRA adapter + processor
trainer.model.save_pretrained(OUT_DIR)   # saves adapter weights/config
processor.save_pretrained(OUT_DIR)

print(f"Saved LoRA adapter + processor to: {OUT_DIR}")