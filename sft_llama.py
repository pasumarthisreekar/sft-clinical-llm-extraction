import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 4096 
dtype = None # Auto-detects bf16 if supported
load_in_4bit = True # Use 4-bit quantization to save VRAM

print("Loading Llama 3.1 8B Instruct...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters
print("Injecting LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 3. Load and Format the Dataset
print("Loading Dataset...")
train_dataset = load_dataset("json", data_files="llama3_sft_dataset_clean.jsonl", split="train")

# Standardize the formatting to Llama 3.1's specific chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)

# 4. Initialize the Trainer
print("Starting Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        warmup_steps = 10,
        num_train_epochs = 3, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Execute Training
trainer_stats = trainer.train()

# 6. Save the trained LoRA adapters
save_path = "llama3_clinical_extraction_lora"
print(f"\nSaving adapter weights to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 7. Export to GGUF format for Ollama
gguf_path = "llama3.1_clinical-unsloth.Q4_K_M.gguf"
print(f"\nExporting to GGUF format for Ollama...")

# This merges the base model and your LoRA weights, quantizes to 4-bit, and creates the .gguf file
model.save_pretrained_gguf(
    "llama3.1_clinical_model", 
    tokenizer, 
    quantization_method = "q4_k_m" 
)

print(f"\nDone! Your fine-tuned model is ready at: llama3.1_clinical_model/unsloth.Q4_K_M.gguf")