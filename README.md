
# HEAL: Harmonizing Efficient Alignment with RLAIF and RLHF for Health AI

HEAL is a a project-in-progress designed to explore and implement reinforcement learning techniques—Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF)—for aligning Large Language Models (LLMs) in healthcare applications.

This repository includes tools for:
- Data preprocessing for healthcare datasets like MIMIC-IV and PubMedQA.
- Supervised fine-tuning (SFT) of policy models.
- Reward model training for RLHF and RLAIF.
- Reinforcement Learning using PPO.
- Evaluation of model robustness and adversarial testing (e.g., Jailbreaks).

## Repository Structure

```
HEAL/
├── data/                     # Data directory
│   ├── processed/            # Preprocessed datasets
│   ├── raw/                  # Raw datasets
├── experiments/              # Experiment configurations, logs, and results
│   ├── configs/              # Training configurations
│   ├── logs/                 # Training logs
│   └── results/              # Evaluation results
├── models/                   # Trained models
│   ├── policy_model/         # Fine-tuned policy models
│   ├── reward_model/         # Trained reward models
├── src/                      # Source code
│   ├── data_preprocessing/   # Data preprocessing scripts
│   ├── evaluation/           # Evaluation scripts
│   ├── training/             # Training scripts (SFT, PPO, reward model training)
│   ├── utils/                # Helper functions
├── environment.yml           # Conda environment setup
├── requirements.txt          # List of dependencies
└── README.md                 # Overview of the project
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/seonokrkim/HEAL.git
cd HEAL
```

### 2. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using Conda (optional):

```bash
conda env create -f environment.yml
conda activate heal
```

## Usage

### 1. Data Preprocessing

Prepare datasets like MIMIC-IV (summarization) and PubMedQA (question-answering).

```bash
# Preprocess MIMIC-IV data
python src/data_preprocessing/preprocess_data.py --dataset mimic \
    --input_path data/raw/mimic_iv_notes.csv \
    --output_path data/processed/mimic_iv_summaries.csv

# Preprocess PubMedQA data
python src/data_preprocessing/preprocess_data.py --dataset pubmedqa \
    --input_path data/raw/pubmedqa.json \
    --output_path data/processed/pubmedqa.csv
```

### 2. Supervised Fine-Tuning (SFT)

Fine-tune a pretrained LLM (e.g., GPT-4) on a processed dataset.

```bash
# Fine-tune on MIMIC-IV
python src/training/sft.py --dataset mimic \
    --dataset_path data/processed/mimic_iv_summaries.csv \
    --model_name gpt-4 \
    --output_dir models/policy_model/mimic_sft

# Fine-tune on PubMedQA
python src/training/sft.py --dataset pubmedqa \
    --dataset_path data/processed/pubmedqa.csv \
    --model_name gpt-4 \
    --output_dir models/policy_model/pubmedqa_sft
```

### 3. Train Reward Models

Train a reward model for RLHF or RLAIF workflows.

```bash
python src/training/reward_model.py --dataset pubmedqa \
    --dataset_path data/processed/pubmedqa.csv \
    --model_name gpt-4 \
    --output_dir models/reward_model/pubmedqa
```

### 4. Reinforcement Learning with PPO

Perform policy optimization using a reward model.

```bash
python src/training/ppo.py --model_name gpt-4 \
    --reward_model_path models/reward_model/pubmedqa \
    --output_dir models/policy_model/pubmedqa_ppo
```

### 5. Evaluation

#### Evaluate Model Performance

```bash
python src/evaluation/evaluate.py --model_path models/policy_model/pubmedqa_sft \
    --dataset_path data/processed/pubmedqa.csv
```

#### Adversarial Testing (e.g., Jailbreaks)

```bash
python src/evaluation/adversarial.py --model_path models/policy_model/pubmedqa_sft \
    --adversarial_prompts_path data/raw/adversarial_prompts.csv
```

## Dependencies

Install the following Python packages:
- `transformers`
- `trl`
- `torch`
- `datasets`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

All dependencies are listed in `requirements.txt`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Work in Progress 🚧

This repository is under active development, with regular updates expected. Stay tuned for new features and improvements!
