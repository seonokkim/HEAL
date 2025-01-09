from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset

# Load dataset
dataset = load_dataset("path_to_ranking_feedback")

# Load reward model
model_name = "gpt-4"
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(
        examples["prompt"] + examples["response"],
        truncation=True,
        padding=True
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training loop
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
for epoch in range(3):
    for batch in tokenized_dataset["train"]:
        inputs = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
        labels = torch.tensor(batch["labels"]).to("cuda")
        outputs = reward_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()