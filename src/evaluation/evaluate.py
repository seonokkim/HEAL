import pandas as pd
import argparse

def preprocess_mimic(input_path, output_path):
    # Load raw MIMIC-IV notes
    mimic_notes = pd.read_csv(input_path)

    # Filter for discharge summaries or ICU notes
    filtered_notes = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"]

    # Clean and preprocess text
    def clean_text(text):
        return text.replace("\n", " ").strip()

    filtered_notes["TEXT"] = filtered_notes["TEXT"].apply(clean_text)

    # Save processed dataset
    filtered_notes.to_csv(output_path, index=False)
    print(f"Processed MIMIC-IV data saved to {output_path}")

def preprocess_pubmedqa(input_path, output_path):
    # Load raw PubMedQA dataset
    pubmedqa_data = pd.read_json(input_path)

    # Select necessary fields
    qa_data = pubmedqa_data[["question", "context", "final_decision"]]

    # Clean text
    def clean_text(text):
        return text.strip()

    qa_data["question"] = qa_data["question"].apply(clean_text)
    qa_data["context"] = qa_data["context"].apply(clean_text)

    # Save processed dataset
    qa_data.to_csv(output_path, index=False)
    print(f"Processed PubMedQA data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for training")
    parser.add_argument("--dataset", type=str, required=True, choices=["mimic", "pubmedqa"],
                        help="Dataset to preprocess: mimic or pubmedqa")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset")

    args = parser.parse_args()

    if args.dataset == "mimic":
        preprocess_mimic(args.input_path, args.output_path)
    elif args.dataset == "pubmedqa":
        preprocess_pubmedqa(args.input_path, args.output_path)