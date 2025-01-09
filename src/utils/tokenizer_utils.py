from transformers import AutoTokenizer

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_texts(tokenizer, texts, max_length=512):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def tokenize_dataset(tokenizer, dataset, text_column="TEXT"):
    return dataset.map(
        lambda x: tokenizer(x[text_column], truncation=True, padding=True), 
        batched=True
    )