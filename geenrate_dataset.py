import json
import os
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Verify the key exists in the environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found. Please set it in your terminal first.")
    exit(1)

# 2. Initialize the modern Gemini Client
client = genai.Client()

class Medication(BaseModel):
    name: str = Field(description="The generic or brand name of the medication")
    dose: str | None = Field(description="The dosage amount and frequency. Return null if missing.")

class ClinicalExtraction(BaseModel):
    primary_diagnoses: list[str] = Field(
        description="A list of primary diagnoses. Complex conditions must be physically split into separate, atomic strings."
    )
    medications: list[Medication]

def process_dataset(input_filepath: str, output_filepath: str, file_mode: str, tail_count: int = None):
    print(f"\nReading {input_filepath}...")

    with open(input_filepath, "r", encoding="utf-8") as infile:
        lines = [line for line in infile if line.strip()]

    # Slice the exact number of lines if requested
    if tail_count:
        lines_to_process = lines[-tail_count:]
        print(f" -> Slicing the last {tail_count} records out of {len(lines)}.")
    else:
        lines_to_process = lines
        print(f" -> Processing all {len(lines)} records.")

    # Open the file safely ("a" for append, "w" for write fresh)
    with open(output_filepath, file_mode, encoding="utf-8") as outfile:
        for idx, line in enumerate(lines_to_process, start=1):
            data = json.loads(line)
            note_id = data.get("note_id", "Unknown")
            patient_text = data.get("clean_text", "")

            print(f" -> [{idx}/{len(lines_to_process)}] Hitting Gemini 3.1 Pro for Note ID: {note_id}...")

            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=patient_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClinicalExtraction,
                        temperature=0.0
                    )
                )

                json_response = response.text

                sft_record = {
                    "messages": [
                        {"role": "system", "content": "Extract the primary diagnoses and medications into structured JSON. Split complex conditions into separate strings."},
                        {"role": "user", "content": patient_text},
                        {"role": "assistant", "content": json_response}
                    ]
                }

                outfile.write(json.dumps(sft_record) + "\n")
                
                # Small pause to ensure we don't trip the Requests-Per-Minute limit
                time.sleep(2)

            except Exception as e:
                print(f"    [!] Error processing Note ID {note_id}: {e}")
                
                # Immediately stop if we hit the limit
                if "429" in str(e):
                    print("    [!] Rate limit hit! Stopping immediately.")
                    return

    print(f"Finished processing and saved to {output_filepath}")

if __name__ == "__main__":
    # 1. Process exactly the last 150 of train data and APPEND ("a")
    process_dataset(
        input_filepath="data/processed/train_notes.jsonl", 
        output_filepath="llama3_sft_dataset.jsonl",
        file_mode="a",      # Append so your 217 existing rows are safe
        tail_count=150      # Only process the final 150 lines
    )

    # 2. Process ALL 100 of test data and WRITE ("w")
    process_dataset(
        input_filepath="data/processed/eval_notes.jsonl", 
        output_filepath="llama3_sft_test_dataset.jsonl",
        file_mode="w",      # Write a fresh file for evaluation
        tail_count=None     # Do all 100 lines
    )