import json
import requests
import time

def get_rxcui(drug_name: str) -> str:
    """
    Queries the NIH RxNav API. Tries exact match first, 
    then falls back to fuzzy/approximate matching for messy strings.
    """
    # --- STEP 1: Try Exact Match ---
    exact_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    
    try:
        response = requests.get(exact_url, params={"name": drug_name})
        response.raise_for_status()
        data = response.json()
        
        if "idGroup" in data and "rxnormId" in data["idGroup"]:
            return data["idGroup"]["rxnormId"][0]
            
    except requests.exceptions.RequestException as e:
        print(f"Exact match API Error for '{drug_name}': {e}")

    # --- STEP 2: Fallback to Approximate Match ---
    approx_url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    
    try:
        # maxEntries=1 ensures we only get the highest confidence fuzzy match
        response = requests.get(approx_url, params={"term": drug_name, "maxEntries": 1})
        response.raise_for_status()
        data = response.json()
        
        if "approximateGroup" in data and "candidate" in data["approximateGroup"]:
            # The approximate endpoint returns the ID inside a candidate list
            return data["approximateGroup"]["candidate"][0]["rxcui"]
            
    except requests.exceptions.RequestException as e:
        print(f"Approximate match API Error for '{drug_name}': {e}")
        
    # If both fail, return None
    return None

def run_mapping():
    input_path = "data/processed/extracted_entities.json"
    output_path = "data/processed/enriched_entities.json"
    
    print(f"Loading Day 2 extractions from {input_path}...")
    
    with open(input_path, "r") as f:
        clinical_data = json.load(f)
        
    mapped_count = 0
    total_meds = 0
    
    for note in clinical_data:
        print(f"\nProcessing Note ID: {note['note_id']}")
        
        # Iterate through the medications extracted by Llama-3
        for med in note['extraction']['medications']:
            total_meds += 1
            drug_name = med['name']
            
            print(f"  -> Looking up RxCUI for: '{drug_name}'...")
            rxcui = get_rxcui(drug_name)
            
            # Attach the standard code to our dataset
            med['rxcui'] = rxcui
            
            if rxcui:
                print(f"     Found: {rxcui}")
                mapped_count += 1
            else:
                print("     No match found.")
                
            # Sleep for 100ms to respect NIH public API rate limits (20 requests/sec)
            time.sleep(0.1)

    # Save the newly enriched dataset
    with open(output_path, "w") as f:
        json.dump(clinical_data, f, indent=2)
        
    print(f"\nMapping complete! Successfully mapped {mapped_count}/{total_meds} medications.")
    print(f"Saved enriched data to {output_path}")

if __name__ == "__main__":
    run_mapping()