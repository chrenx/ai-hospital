
"""
Convert clinical descriptions to short, symptom-only layman language
using GPT-4o (custom endpoint).

• Input : datasets/mtsamples_selected_medicine_cleaned.csv
• Output: datasets/mtsamples_layman_v2.csv         (only successful rows)
          datasets/mtsamples_missing.csv           (rows that failed or returned no clear symptoms)
"""

import sys, re
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# ─────────────── LLM CONFIG ──────────────────────────────────────
llm_config = {
    "config_list": [
        {
            "model":    "gpt-4o",
            "base_url": "https://api.ai.it.ufl.edu/v1",
            "api_key":  "sk-JoOzBJv4uLYVjOKFU5if1w",  # Replace with your real key
        }
    ]
}
_cfg  = llm_config["config_list"][0]
MODEL = _cfg["model"]

client = OpenAI(api_key=_cfg["api_key"], base_url=_cfg["base_url"])
# ────────────────────────────────────────────────────────────────

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

SYMPTOM_LIKE_REGEX = re.compile(r'\b(pain|ache|feel|tired|itch|burn|pressure|weak|nausea|dizzy|sore|vomit|swell|trouble|can\'t|cough|bleed|tight|choke|shortness|difficulty|breathe|rash|loss|cramp|faint|numb|tingle|blur|stuffy|congest|throbbing|sharp)\b', re.I)

def generate_layman(description: str, transcription: str, specialty: str, timeout: int = 60) -> Optional[str]:
    """Generate a layman rewrite from GPT-4o, or return None on failure or invalid output."""
    prompt_input = f"Specialty: {specialty}\n\n{description}\n\n{transcription}".strip()

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            temperature=0.7,
            max_tokens=256,
            timeout=timeout,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_input},
            ],
        )
        text = rsp.choices[0].message.content.strip()

        # Reject if it's empty or lacks any patient-perceived symptoms
        if len(text) < 5 or not SYMPTOM_LIKE_REGEX.search(text):
            return None

        return text
    except (OpenAIError, Exception) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return None

def main() -> None:
    IN_CSV   = Path("datasets/mtsamples_selected_medicine_cleaned.csv")
    OUT_CSV  = Path("datasets/mtsamples_layman_v3.csv")
    MISS_CSV = Path("datasets/mtsamples_missing.csv")

    df_in = pd.read_csv(IN_CSV)
    good_rows, missing_rows = [], []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Rewriting"):
        desc = row["description"]
        # Handle non-string (nan, float, etc)
        if not isinstance(desc, str):
            desc = "" if pd.isna(desc) else str(desc)
        desc_word_count = len(desc.strip().split())

        if desc_word_count < 10:
            missing_rows.append({
                "row_id"       : row["row_id"],
      
                "original_description"  : row["description"],
                "transcription": row["transcription"],
                "medical_specialty": row["medical_specialty"],
            })
            continue 

        layman = generate_layman(row["description"], row["transcription"], row["medical_specialty"])
        if layman:
            layman = layman.strip('\'"')
            good_rows.append({
                "row_id"             : row["row_id"],
                "description" : layman,
                "medical_specialty"  : row["medical_specialty"],
                "sample_name"        : row["sample_name"],
                "keywords"           : row["keywords"],
                "original_description": row["description"],
            })
        else:
            missing_rows.append({
                "row_id"       : row["row_id"],
      
                "original_description"  : row["description"],
                "transcription": row["transcription"],
                "medical_specialty": row["medical_specialty"],
            })

    if good_rows:
        pd.DataFrame(good_rows).to_csv(OUT_CSV, index=False)
        print(f"✅ Saved {len(good_rows)} rows to {OUT_CSV}")
    else:
        print("❌ No successful rows written.")

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(MISS_CSV, index=False)
        print(f"⚠️  {len(missing_rows)} rows had no valid output. See {MISS_CSV}")

if __name__ == "__main__":
    main()


