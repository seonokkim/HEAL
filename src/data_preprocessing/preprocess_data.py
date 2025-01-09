import pandas as pd
import os

def preprocess_mimic(input_path, output_path):
    mimic_notes = pd.read_csv(input_path)
    filtered_notes = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"]
    filtered_notes["TEXT"] = filtered_notes["TEXT"].str.replace("\n", " ").str.strip()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_notes.to_csv(output_path, index=False)
    print(f"MIMIC-IV data processed and saved to {output_path}")

def preprocess_pubmedqa(input_path, output_path):
    pubmedqa_data = pd.read_json(input_path)
    qa_data = pubmedqa_data[["question", "context", "final_decision"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    qa_data.to_csv(output_path, index=False)
    print(f"PubMedQA data processed and saved to {output_path}")

def preprocess_dataset(dataset_name, input_path, output_path):
    if dataset_name == "mimic":
        preprocess_mimic(input_path, output_path)
    elif dataset_name == "pubmedqa":
        preprocess_pubmedqa(input_path, output_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")