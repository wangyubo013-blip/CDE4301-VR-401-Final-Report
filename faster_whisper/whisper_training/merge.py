from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

BASE_MODEL = "/root/autodl-tmp/whisper_project/whisper_training/models/whisper-large-v3"          # your original HF Whisper model
LORA_DIR = "/root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-lora-continued-3/checkpoint-100"    # folder in your screenshot
MERGED_DIR = "/root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-merged-continued-3"

# Load base model and processor locally
processor = WhisperProcessor.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    use_safetensors=True
)

# Attach LoRA
model = PeftModel.from_pretrained(base_model, LORA_DIR)

# Merge LoRA into base weights
merged_model = model.merge_and_unload()

# Save merged full model
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
processor.save_pretrained(MERGED_DIR)

print(f"Merged model saved to: {MERGED_DIR}")

'''
cd /root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-merged-continued-3
cp processor_config.json preprocessor_config.json 
cd ..
ct2-transformers-converter \
  --model /root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-merged-continued-3 \
  --output_dir /root/autodl-tmp/whisper_project/whisper_training/whisper-large-v3-atcosim-ct2-continued-3 \
  --copy_files tokenizer.json processor_config.json \
  --quantization float16
'''