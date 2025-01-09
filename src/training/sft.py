from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import argparse

from utils.tokenizer_utils import load_tokenizer, tokenize_dataset


def fine_tune_sft(dataset_name, dataset_path, model_name, output_dir, num_labels=None):
    # Log dataset being used
    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset("csv", data_files=dataset_path)
    tokenizer = load_tokenizer(model_name)

    # Tokenize dataset
    if dataset_name == "mimic":
        tokenized_dataset = tokenize_dataset(tokenizer, dataset, text_column="TEXT")
    elif dataset_name == "pubmedqa":
        tokenized_dataset = tokenize_dataset(tokenizer, dataset, text_column="question")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load pretrained LLM
    if num_labels:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    print(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on specific datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "pubmedqa"],
                        help="Dataset to train on: mimic or pubmedqa")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (CSV)")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., gpt-4)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")

    args = parser.parse_args()

    if args.dataset == "mimic":
        fine_tune_sft(args.dataset, args.dataset_path, args.model_name, args.output_dir)
    elif args.dataset == "pubmedqa":
        fine_tune_sft(args.dataset, args.dataset_path, args.model_name, args.output_dir, num_labels=3)