import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def ingest_pmc_patients():
    logging.info("Downloading PMC-Patients dataset from Hugging Face...")
    
    # Load the open-access dataset (this will cache locally)
    dataset = load_dataset("zhengyun21/PMC-Patients", split="train")
    
    # Convert to Pandas and take a 500-note subset for our fast 1-week project scope
    df = dataset.to_pandas().head(500)
    logging.info(f"Loaded {len(df)} real clinical case summaries.")

    # The dataset stores the clinical text in a column named 'patient'.
    # We rename it to 'clean_text' and map 'patient_uid' to 'note_id' 
    # so it perfectly matches the Day 2 extraction script we already wrote.
    df = df.rename(columns={'patient': 'clean_text', 'patient_uid': 'note_id'})
    
    # Drop unnecessary metadata columns to save space
    df = df[['note_id', 'clean_text']]

    # Train/Eval Split (80/20 split -> 400 Train, 100 Eval)
    logging.info("Splitting data into Training (400) and Evaluation (100) sets...")
    train_df, eval_df = train_test_split(df, test_size=0.20, random_state=42)

    # Export to JSONL just like before
    train_path = PROCESSED_DIR / "train_notes.jsonl"
    eval_path = PROCESSED_DIR / "eval_notes.jsonl"
    
    train_df.to_json(train_path, orient="records", lines=True)
    eval_df.to_json(eval_path, orient="records", lines=True)
    
    logging.info("Data saved successfully. Ready for Day 2 LLM Extraction.")

if __name__ == "__main__":
    ingest_pmc_patients()
