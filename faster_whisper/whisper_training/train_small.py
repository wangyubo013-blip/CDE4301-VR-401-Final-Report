import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import PeftModel

from datasets import load_from_disk

processed_ds = load_from_disk(
    "/root/autodl-tmp/whisper_project/whisper_training/processed_dataset-03"
)

print(processed_ds)

BASE_MODEL = "/root/autodl-tmp/whisper_project/whisper_training/models/whisper-large-v3"
CKPT_DIR = "/root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-lora-continued-2/checkpoint-100"
OUT_DIR = "/root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-lora-continued-3"

processor = WhisperProcessor.from_pretrained(BASE_MODEL, local_files_only=True)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    use_safetensors=True
)

base_model.generation_config.language = "en"
base_model.generation_config.task = "transcribe"
base_model.generation_config.forced_decoder_ids = None
base_model.generation_config.suppress_tokens = []

model = PeftModel.from_pretrained(
    base_model,
    CKPT_DIR,
    is_trainable=True
)

model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_steps=100,
    logging_steps=5,
    save_steps=25,
    gradient_checkpointing=True,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_ds,
    data_collator=data_collator,
)

trainer.train(

)