import json
import os
import re

import re

def is_verbose_sentence(text):
    """
    Safely checks if a string is a hallucinated English sentence.
    Will NOT delete complex tapering/titration/cyclic schedules.
    """
    if not text or not isinstance(text, str):
        return False
    
    # 1. If the string contains ANY digits, trust the LLM. 
    # It is likely a valid complex dose (e.g., "5 x 500 mg capsule for 14 days...")
    if bool(re.search(r'\d', text)):
        return False
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # 2. If it has no digits, but contains valid text-only frequencies, keep it.
    # (e.g., "twice daily", "every morning", "prn", "as needed", "continuous infusion")
    valid_text_frequencies = [
        # Spelled-out numbers (crucial since \d misses them)
        "one", "two", "three", "four", "half", "quarter", "single", "double",
        
        # Standard English frequencies
        "daily", "twice", "thrice", "times", "every", "weekly", "monthly", "hourly",
        "alternate", "alternating", "continuous", "continuously",
        
        # Latin & common medical abbreviations (your script converts to lower case, so these work perfectly)
        "bid", "tid", "qid", "qds", "qod", "od", "prn", "stat", "hs", "am", "pm",
        
        # Clinical actions & treatment structures
        "course", "cycles", "cycle", "taper", "tapered", "titrate", "titrated",
        "maintenance", "loading", "sliding", "scale",
        
        # Delivery mechanisms
        "bolus", "infusion", "drip", "push", "drop", "drops",
        
        # Time of day / Triggers
        "morning", "night", "bedtime", "evening", "meals", "food", "needed", "immediately", "now"
    ]
    if any(word in text_lower for word in valid_text_frequencies):
        # Only flag if it is absurdly long (over 12 words) with no numbers
        if word_count > 12:
            return True
        return False
        
    # 3. If it has NO digits, NO frequency words, and is longer than 5 words,
    # it is almost certainly a hallucinated conversational sentence.
    if word_count > 5:
        return True
        
    return False

def normalize_whitespace_and_schema(input_filepath, output_filepath):
    if not os.path.exists(input_filepath):
        print(f"[!] Skipping {input_filepath} - File not found.")
        return

    print(f"Cleaning {input_filepath}...")
    cleaned_count = 0
    modified_doses_count = 0
    error_count = 0

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            
            try:
                # 1. Parse the outer ChatML JSONL line
                record = json.loads(line)
                
                # 2. Find the assistant's response in the messages array
                for message in record.get("messages", []):
                    if message.get("role") == "assistant":
                        
                        raw_inner_string = message["content"]
                        inner_dict = json.loads(raw_inner_string)
                        
                        # 3. Clean up the 'dose' fields
                        if "medications" in inner_dict:
                            for med in inner_dict["medications"]:
                                dose_value = med.get("dose")
                                
                                # A. Check for verbose English sentences
                                if is_verbose_sentence(dose_value):
                                    med["dose"] = None
                                    modified_doses_count += 1
                                
                                # B. Set explicitly empty strings to None (null)
                                elif dose_value == "":
                                    med["dose"] = None
                        
                        # 4. Convert it back to a strictly minified string with ZERO spaces
                        minified_inner_string = json.dumps(inner_dict, separators=(',', ':'))
                        message["content"] = minified_inner_string
                
                # 5. Write the entirely minified record back to the new file
                outfile.write(json.dumps(record, separators=(',', ':')) + "\n")
                cleaned_count += 1
                
            except json.JSONDecodeError as e:
                print(f"    [!] JSON parsing error on line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"    [!] Unexpected error on line {line_num}: {e}")
                error_count += 1

    print(f" -> Successfully cleaned: {cleaned_count} records.")
    print(f" -> Verbose sentences converted to null: {modified_doses_count}")
    if error_count > 0:
        print(f" -> Errors encountered: {error_count}")
    print(f" -> Saved perfect data to: {output_filepath}\n")

if __name__ == "__main__":
    # Clean the training dataset
    normalize_whitespace_and_schema(
        input_filepath="llama3_sft_dataset.jsonl", 
        output_filepath="llama3_sft_dataset_clean.jsonl"
    )

    # Clean the evaluation dataset
    normalize_whitespace_and_schema(
        input_filepath="llama3_sft_test_dataset.jsonl", 
        output_filepath="llama3_sft_test_dataset_clean.jsonl"
    )