import logging
import os

def setup_logging(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        filename=log_file_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    return logging.getLogger()

def log_metrics(logger, metrics, step):
    for key, value in metrics.items():
        logger.info(f"Step {step}: {key} = {value:.4f}")