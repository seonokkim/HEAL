from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def evaluate_adversarial(model_path, adversarial_prompts_path):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load adversarial prompts
    adversarial_prompts = pd.read_csv(adversarial_prompts_path)

    for _, row in adversarial_prompts.iterrows():
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Adversarial Prompt: {prompt}\nModel Response: {response}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on adversarial prompts")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--adversarial_prompts_path", type=str, required=True, help="Path to adversarial prompts CSV file")
    args = parser.parse_args()

    evaluate_adversarial(args.model_path, args.adversarial_prompts_path)