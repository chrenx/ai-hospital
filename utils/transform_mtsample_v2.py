


"""
Convert clinical descriptions to short, symptom-only layman language
using GPT-4o (custom endpoint).  Rows that fail or exceed 60 s are
logged to mtsamples_missing.csv.

Input : datasets/mtsamples_selected_medicine_cleaned.csv
Output: datasets/mtsamples_layman_v2.csv
        datasets/mtsamples_missing.csv
"""

# from __future__ import annotations
import time, sys, csv
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# ─────────────────── LLM CONFIG ──────────────────────────────────────────────
llm_config = {
    "config_list": [
        {
            "model":    "gpt-4o",
            "base_url": "https://api.ai.it.ufl.edu/v1",
            "api_key":  "sk-JoOzBJv4uLYVjOKFU5if1w",
        }
    ]
}
_cfg  = llm_config["config_list"][0]
MODEL = _cfg["model"]

client = OpenAI(
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"]
)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Rewrite the text below so it sounds like the patient is briefly describing "
    "only what they FEEL right now, **before** any doctor visit.\n"
    "• Keep it in plain, everyday language.\n"
    "• Use 1-3 short sentences (max 40 words).\n"
    "• Mention ONLY what the patient can personally feel or notice (e.g., pain, fatigue, coughing).\n"
    "• DO NOT mention diseases, diagnoses, history, meds, tests, or hospitals.\n"
    "• If no clear symptoms are present, you may infer 1-2 common sensations based on the specialty (e.g., blurry vision for Ophthalmology).\n"
    "• If even that is not possible, return NOTHING."
)

def generate_layman(description: str, transcription: str, timeout: int = 60) -> Optional[str]:
    """
    Ask GPT-4o for a symptom-only rewrite.
    Returns None on timeout or any exception.
    """
    text = f"{description}\n\n{transcription}".strip()

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            temperature=0.7,
            max_tokens=1024,
            timeout=timeout,           # hard 60-s cut-off
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
        )
        return rsp.choices[0].message.content.strip()
    except OpenAIError as e:
        # includes timeouts, API errors, etc.
        print(f"[GPT-4o error] {e}", file=sys.stderr)
    except Exception as e:
        print(f"[Unexpected error] {e}", file=sys.stderr)
    return None

def main() -> None:
    IN_CSV  = Path("datasets/mtsamples_selected_medicine_cleaned.csv")
    OUT_CSV = Path("datasets/mtsamples_layman_v2.csv")
    MISS_CSV = Path("datasets/mtsamples_missing.csv")

    df_in  = pd.read_csv(IN_CSV)
    good_rows, missing_rows = [], []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Rewriting"):
        layman = generate_layman(row["description"], row["transcription"])

        # Build row for main CSV
        out_row: Dict[str, Any] = {
            "row_id": row["row_id"],
            "description": layman if layman else row["description"],  # fall back to original
            "medical_specialty": row["medical_specialty"],
            "sample_name": row["sample_name"],
            "keywords": row["keywords"],
            "original_description": row["description"],               # last column
        }
        good_rows.append(out_row)

        # Track failures
        if layman is None:
            missing_rows.append({
                "row_id": row["row_id"],
                "description": row["description"],
                "transcription": row["transcription"],
            })

    # Write main output
    pd.DataFrame(good_rows).to_csv(OUT_CSV, index=False)

    # Write missing rows (if any)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(MISS_CSV, index=False)
        print(f"⚠️  {len(missing_rows)} rows timed-out/failed. See {MISS_CSV}")

    print(f"✅ Finished. Output saved to {OUT_CSV}")

if __name__ == "__main__":
    main()