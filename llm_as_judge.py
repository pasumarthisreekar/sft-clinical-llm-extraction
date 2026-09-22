import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

# 1. Define the exact JSON structure you want Gemini to return
class EvaluationResult(BaseModel):
    reasoning: str
    preferred_model: str  # We will instruct it to return "Base", "Finetuned", or "Tie"

def run_llm_judge(dataset):
    # Initialize the modern SDK client (assumes GEMINI_API_KEY is in your environment)
    client = genai.Client()
    
    scoreboard = {"base_wins": 0, "finetuned_wins": 0, "ties": 0}
    
    for idx, item in enumerate(dataset):
        # 2. Build a strict evaluation prompt
        prompt = f"""
        You are an expert AI evaluator. Compare the performance of two models against the Ground Truth.
        
        [Input Prompt]: {item['input']}
        [Ground Truth]: {item['ground_truth']}
        
        [Model A - Base]: {item['base_output']}
        [Model B - Finetuned]: {item['finetuned_output']}
        
        Analyze which model's output is more accurate, complete, and aligned with the Ground Truth.
        Return your step-by-step reasoning, and then declare the preferred_model as strictly "Base", "Finetuned", or "Tie".
        """
        
        # 3. Call Gemini Pro with Structured Outputs
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResult,
                temperature=0.1, # Keep temperature low for deterministic evaluations
            ),
        )
        
        # 4. Parse the JSON and update the scoreboard
        try:
            eval_data = json.loads(response.text)
            winner = eval_data.get("preferred_model", "").lower()
            
            if "base" in winner:
                scoreboard["base_wins"] += 1
            elif "finetuned" in winner:
                scoreboard["finetuned_wins"] += 1
            else:
                scoreboard["ties"] += 1
                
            print(f"Sample {idx+1}: Judge picked {winner.upper()}")
            
        except Exception as e:
            print(f"Error parsing sample {idx+1}: {e}")
            
    return scoreboard

# --- Example Execution ---
if __name__ == "__main__":
    # Replace this with your actual data loaded from a CSV or JSON file
    sample_data = [
        {
            "input": "What is the capital of France?",
            "ground_truth": "Paris",
            "base_output": "I think it is Paris.",
            "finetuned_output": "Paris."
        }
    ]
    
    final_scores = run_llm_judge(sample_data)
    print("\nFinal Results:", final_scores)