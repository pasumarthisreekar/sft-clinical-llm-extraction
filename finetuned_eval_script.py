import json
import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
import ollama

# 1. Define the strict JSON schema target
class Medication(BaseModel):
    name: str = Field(description="The generic or brand name of the medication")
    dose: Optional[str] = Field(description="The dosage amount and frequency, if specified")

class ClinicalExtraction(BaseModel):
    primary_diagnoses: List[str] = Field(
        description="A list of all primary diagnoses. If a diagnosis contains multiple conditions (e.g., condition A complicated by condition B), split them into separate, atomic strings."
    )
    medications: List[Medication] = Field(description="List of all active medications mentioned")

def extract_clinical_data(text: str) -> str:
    """Sends clinical text to your fine-tuned local Llama with safety limits."""
    prompt = f"Extract the primary diagnoses (splitting complex conditions into separate items) and medications from this discharge summary:\n\n{text}"
    
    try:
        response = ollama.chat(
            model='clinical_llama',  # <--- UPDATED to your fine-tuned model
            messages=[{'role': 'user', 'content': prompt}],
            format=ClinicalExtraction.model_json_schema(),
            options={
                'temperature': 0.0,    # Zero randomness prevents looping/stuttering
                'num_predict': 2048    # Safe cap to prevent infinite generation freezes
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"  -> [!] Ollama API Error: {e}")
        return "{}"

def run_extraction():
    print("Loading Evaluation Data (JSONL) - Restricted to first 5 entries...")
    # .head(5) limits the dataset to just the first 5 rows
    df = pd.read_json("data/processed/eval_notes.jsonl", orient="records", lines=True).head(100)
    total_notes = len(df)
    
    # Load existing results to support seamless resuming if it crashed earlier
    output_path = "finetuned_eval_results.json"  # <--- UPDATED output file name
    results = []
    processed_ids = set()
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            processed_ids = {item["note_id"] for item in results}
        print(f"Resuming... Found {len(processed_ids)} already processed notes. Skipping them.")
    
    for idx, row in df.iterrows():
        note_id = row['note_id']
        if note_id in processed_ids:
            continue
            
        current_step = idx + 1
        print(f"Processing Note ID: {note_id} [{current_step}/{total_notes}]...")
        json_output = extract_clinical_data(row['clean_text'])
        
        # Validate the LLM output against the Pydantic schema
        try:
            validated_data = ClinicalExtraction.model_validate_json(json_output)
            results.append({
                "note_id": note_id,
                "status": "success",
                "extraction": validated_data.model_dump()
            })
            print(f"  -> Successfully extracted {len(validated_data.primary_diagnoses)} diagnoses and {len(validated_data.medications)} medications.")
            
        except Exception as e:
            print(f"  -> [!] Validation failed for Note {note_id}: {e}")
            results.append({
                "note_id": note_id,
                "status": "failed",
                "error": str(e),
                "raw_output": json_output
            })
            
        # Periodically save progress after every single note
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
    print(f"\nFinished! Saved all fine-tuned extractions to {output_path}")

if __name__ == "__main__":
    run_extraction()