from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
import os

# Optimized Configuration for Speed
MODEL_NAME = "google/flan-t5-small"  # Smaller model for faster training
DATASET_NAME = "squad"
OUTPUT_DIR = "models/quest_generator"
MAX_LENGTH = 128                     # Reduced for speed
BATCH_SIZE = 16                      # Increased batch size
EPOCHS = 1                           # Single epoch
TRAIN_SAMPLES = 5000                 # Limited training samples

def main():
    # Load and limit dataset for speed
    dataset = load_dataset(DATASET_NAME)
    
    # Limit training samples for faster training
    train_dataset = dataset['train'].select(range(min(TRAIN_SAMPLES, len(dataset['train']))))
    eval_dataset = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    
    print(f"✅ Using {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimized preprocessing
    def preprocess_function(examples):
        inputs = [f"question: {context}" for context in examples["context"]]
        targets = examples["question"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False  # Dynamic padding in data collator
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize datasets
    train_tokenized = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4  # Parallel processing
    )
    
    eval_tokenized = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4
    )

    # Load smaller model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()

    # Minimal training arguments for speed
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",              # No evaluation during training
        learning_rate=3e-4,              # Higher LR for faster convergence
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        warmup_steps=100,                # Minimal warmup
        logging_steps=50,                # Less frequent logging
        save_steps=10000,                # Save only at end
        save_total_limit=1,              # Keep only final model
        remove_unused_columns=True,
        dataloader_num_workers=2,        # Faster data loading
        fp16=torch.cuda.is_available(),  # Mixed precision for speed
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=False,    # Skip best model loading
        metric_for_best_model=None,
        greater_is_better=None,
        predict_with_generate=False,     # Skip generation during eval
    )

    # Data collator with dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Trainer setup
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    print("🚀 Starting fast fine-tuning...")
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")

    # Quick test
    print("\n🧪 Quick test:")
    test_context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    
    model.eval()
    inputs = tokenizer(f"question: {test_context}", return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=2,
            early_stopping=True
        )
    
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Context: {test_context}")
    print(f"Generated Question: {question}")

if __name__ == "__main__":
    main()