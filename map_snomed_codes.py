import json
import requests
import time
import re

import os

# Set this in your terminal before running: export UMLS_API_KEY="your_key_here"
UMLS_API_KEY = os.environ.get("UMLS_API_KEY")

def get_snomed_code(diagnosis_str: str):
    """
    Queries the secure NIH UMLS REST API for SNOMED CT codes.
    Requires a UMLS API Key.
    """
    if not UMLS_API_KEY:
        raise ValueError("Missing UMLS_API_KEY environment variable.")

    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    
    # --- STEP 1: UMLS Exact Search ---
    params = {
        "string": diagnosis_str,
        "sabs": "SNOMEDCT_US",        # Restrict search ONLY to the SNOMED dictionary
        "returnIdType": "sourceUi",   # Return the SNOMED code instead of the UMLS CUI
        "searchType": "exact",        # Start with strict matching
        "apiKey": UMLS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("result", {}).get("results", [])
        if results and results[0]["ui"] != "NONE":
            return results[0]["ui"], results[0]["name"]
            
    except requests.exceptions.RequestException as e:
        print(f"UMLS Exact match API Error for '{diagnosis_str}': {e}")

    # --- STEP 2: UMLS Approximate (Fuzzy) Search ---
    print("     Exact match failed. Falling back to approximate search...")
    params["searchType"] = "approximate"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("result", {}).get("results", [])
        if results and results[0]["ui"] != "NONE":
            return results[0]["ui"], results[0]["name"]
            
    except requests.exceptions.RequestException as e:
        print(f"UMLS Approximate match API Error for '{diagnosis_str}': {e}")

    # If both fail, return None
    return None, None
def run_snomed_mapping():
    # Read the data that already has the Day 3 RxNorm codes attached
    input_path = "data/processed/enriched_entities.json"
    output_path = "data/processed/final_clinical_data.json"
    
    print(f"Loading Day 3 enriched data from {input_path}...")
    
    with open(input_path, "r") as f:
        clinical_data = json.load(f)
        
    mapped_count = 0
    total_diagnoses = 0
    
    for note in clinical_data:
        print(f"\nProcessing Note ID: {note['note_id']}")
        
        # We will build a new array of dictionaries to hold the mapped data
        mapped_diagnoses = []
        
        # Iterate through the NEW array of diagnoses from the LLM
        for diagnosis_text in note['extraction']['primary_diagnoses']:
            total_diagnoses += 1
            print(f"  -> Looking up SNOMED for: '{diagnosis_text}'...")
            
            snomed_code, official_name = get_snomed_code(diagnosis_text)
            
            if snomed_code:
                print(f"     Found: {snomed_code} ({official_name})")
                mapped_count += 1
            else:
                print("     No match found.")
                
            # Append the structured object
            mapped_diagnoses.append({
                "raw_text": diagnosis_text,
                "snomed_ct_code": snomed_code,
                "snomed_ct_name": official_name
            })
            
            # Respect NIH rate limits (20 requests/sec)
            time.sleep(0.1) 
            
        # Overwrite the raw list of strings with our new standardized list of objects
        note['extraction']['primary_diagnoses'] = mapped_diagnoses

    # Save the final dataset
    with open(output_path, "w") as f:
        json.dump(clinical_data, f, indent=2)
        
    print(f"\nSNOMED Mapping complete! Successfully mapped {mapped_count}/{total_diagnoses} diagnoses.")
    print(f"Saved final standardized data to {output_path}")

if __name__ == "__main__":
    run_snomed_mapping()