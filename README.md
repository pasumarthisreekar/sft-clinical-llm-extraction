# Fine-Tuned LLM for Clinical Information Extraction

Fine-tunes Llama-3.1-8B-Instruct with LoRA to extract structured diagnoses and medications from clinical notes into strict JSON, and evaluates the result against the base model on real clinical text.

## What it does

- Sources 500 real clinical case summaries from the public **PMC-Patients** dataset
- Distills structured training labels (diagnoses + medications as JSON) using the **Gemini API** with a Pydantic-enforced schema
- Cleans hallucinated/verbose dosage text out of the labels with rule-based heuristics
- Fine-tunes **Llama-3.1-8B-Instruct** with **LoRA** via **Unsloth** (4-bit quantization, rank 16)
- Evaluates base vs. fine-tuned model output against ground truth, plus an **LLM-as-judge** (Gemini) pairwise comparison
- Exports the fine-tuned model to **GGUF** for local inference via **Ollama**

## Results

Evaluated against ground truth on a 50-note holdout set:

| Metric | Base Model | Fine-Tuned Model |
|---|---|---|
| Primary Diagnosis Recall | 38.2% | 73.2% |
| Medication Recall | 77.5% | 89.9% |
| Dosage-Field Accuracy | 48.6% | 96.0% |

## Pipeline

```
PMC-Patients (public dataset)
      │
      ▼
Gemini API (label distillation, Pydantic schema)
      │
      ▼
Data cleaning (remove hallucinated dosage text)
      │
      ▼
LoRA fine-tuning (Unsloth, Llama-3.1-8B-Instruct)
      │
      ▼
Evaluation: exact-match vs. ground truth + LLM-as-judge
      │
      ▼
Export to GGUF → served locally via Ollama
```

## Tech stack

Python, PyTorch, Unsloth, LoRA, Transformers, TRL, Ollama, Pydantic, Gemini API

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
3. Ingest the source dataset:
   ```bash
   python ingest_pmc_patients.py
   ```
4. Generate the training/eval dataset:
   ```bash
   python generate_dataset.py
   ```
5. Clean the dataset:
   ```bash
   python clean_dataset.py
   ```
6. Fine-tune the model:
   ```bash
   python sft_llama.py
   ```
7. Run evaluation on the base and fine-tuned models:
   ```bash
   python extract_baseline_llama3.py       # base model eval
   python extract_finetuned_llama3.py      # fine-tuned model eval
   ```
8. Compare with LLM-as-judge:
   ```bash
   python llm_as_judge.py
   ```
9. (Optional) Map extracted entities to standard medical codes:
   ```bash
   python map_rxnorm_codes.py    # medications -> RxNorm
   python map_snomed_codes.py    # diagnoses -> SNOMED CT
   ```

## Files

- `ingest_pmc_patients.py` — downloads and splits the PMC-Patients dataset into train/eval sets
- `generate_dataset.py` — generates SFT training/eval data via Gemini
- `clean_dataset.py` — cleans hallucinated dosage annotations from labels
- `sft_llama.py` — LoRA fine-tuning script (Unsloth)
- `extract_baseline_llama3.py` — runs the base (un-tuned) model on eval notes
- `extract_finetuned_llama3.py` — runs the fine-tuned model on eval notes
- `llm_as_judge.py` — pairwise LLM-judged comparison of base vs. fine-tuned outputs
- `map_rxnorm_codes.py` — links extracted medications to RxNorm codes (NIH RxNav API)
- `map_snomed_codes.py` — links extracted diagnoses to SNOMED CT codes (NIH UMLS API)
- `Modelfile` — Ollama model definition for local serving

## Notes

This project uses the publicly available PMC-Patients dataset as its data source. No real/private patient data is used.
