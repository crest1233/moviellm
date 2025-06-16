from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
import torch

# Load dataset
dataset = load_dataset("json", data_files="movie_descriptions.jsonl", split="train")

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Prepare model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Tokenize
def tokenize(example):
    prompt = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']} </s>"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Trainer
training_args = TrainingArguments(
    output_dir="./finetuned_llama_movie",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=50,
    logging_dir="./logs",
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=2
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()
