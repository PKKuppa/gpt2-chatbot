
#%%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

#%%
# Load and tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Load the formatted chat data
def load_chat_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n\n")  # Splitting conversations
    return Dataset.from_dict({"text": lines})

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load and tokenize dataset
dataset = load_chat_dataset("../data/gpt2_train.txt")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#%%
# Define model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to("cuda")

# Define training arguments
training_args = TrainingArguments(
    output_dir="../model/gpt2",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,  # Save checkpoints every 500 steps
    save_total_limit=2,  # Keep only the latest 2 checkpoints
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="no",  # No validation dataset
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=True,  # Enable mixed precision if on GPU
    learning_rate=1e-5,
    per_device_eval_batch_size=8,
)

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save final model
model.save_pretrained("../model/gpt2_final")
tokenizer.save_pretrained("../model/gpt2_final")


# %%
