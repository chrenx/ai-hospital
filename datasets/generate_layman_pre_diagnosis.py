
"""
Rewrite the 'description' field in mtsamples.csv so it sounds like
a patient's own words, before any diagnosis or treatment is known.

- Uses the custom GPT-4o endpoint provided in llm_config
- Retries for up to 5 min per row before giving up
- Keeps original text in a new 'original_description' column
- Logs missing rewrites in mtsamples_missing.txt
"""
from __future__ import annotations
import csv, os, sys, time
from datetime import timedelta
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

# ───── CONFIG ────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("mtsamples_layman.csv")
OUTPUT_CSV = Path("mtsamples_pre_diagnosis.csv")
MISSING_TXT = Path("mtsamples_missing.txt")

llm_config = {                 #  ← ⬅︎ your format
    "config_list": [
        {
            "model":    "gpt-4o",
            "base_url": "https://api.ai.it.ufl.edu/v1",
            "api_key":  "sk-JoOzBJv4uLYVjOKFU5if1w",
        }
    ]
}

TEMPERATURE = 0.7
MAX_TOKENS  = 1024
TIME_LIMIT_PER_ROW = timedelta(minutes=2)
# ─────────────────────────────────────────────────────────────────────────────

# pick first (only) config entry
_cfg = llm_config["config_list"][0]
MODEL = _cfg["model"]


# SYSTEM_PROMPT = (
#     "You are an expert medical writer. Rewrite the following clinical note entirely "
#     "in plain, conversational language as if the PATIENT is casually describing their "
#     "own symptoms and discomfort before seeing a doctor. Do not include any medical "
#     "terms, diagnoses, test results, or treatments — remove all of that. "
#     "Only describe what the patient can personally feel or observe, like pain, fatigue, "
#     "swelling, etc. Keep it brief and natural, like something an average person would say "
#     "when explaining how they feel — avoid long or overly detailed explanations."
# )
SYSTEM_PROMPT = (
    "Rewrite the text below so it sounds like the patient is briefly describing "
    "only what they FEEL right now, **before** any doctor visit. "
    "• Keep it in plain, everyday language. "
    "• Use 1-3 short sentences (max ~40 words total). "
    "• Mention ONLY current sensations or discomfort the patient can personally notice "
    "   (e.g., sneezing, itchy eyes, stuffy nose, tiredness). "
    "• Remove all medication names, treatments tried, diagnoses, medical history, "
    "  locations they have lived, demographic details, and anything the patient "
    "couldn't know without a doctor. "
    "Return ONLY the rewritten text."
)

client = OpenAI(
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"]
)

# ───── Helpers ───────────────────────────────────────────────────────────────
def call_llm(original: str, deadline: float) -> Optional[str]:
    attempt = 0
    while time.time() < deadline:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Original note:\n\"\"\"\n{original}\n\"\"\"\n\nRewrite:"},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            attempt += 1
            wait = min(2**attempt, 30)
            if time.time() + wait > deadline:
                break
            print(f"[retry in {wait}s] {e}", file=sys.stderr)
            time.sleep(wait)
    return None
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_CSV.exists():
        sys.exit(f"❌ {INPUT_CSV} not found")

    failed_rows: list[str] = []

    with INPUT_CSV.open(newline='', encoding="utf-8") as fin, \
        OUTPUT_CSV.open("w", newline='', encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["original_description"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Rewriting"):
            original = row["description"]
            row["original_description"] = original

            deadline = time.time() + TIME_LIMIT_PER_ROW.total_seconds()
            rewritten = call_llm(original, deadline)

            if rewritten is None:
                # keep original text in place; flag the miss
                failed_rows.append(row.get("row_id", "<no row_id>"))
                rewritten = original  # or "" if you prefer blanks

            row["description"] = rewritten
            writer.writerow(row)
            break #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if failed_rows:
        with MISSING_TXT.open("w", encoding="utf-8") as f:
            f.write("\n".join(failed_rows))
        print(f"⚠️  {len(failed_rows)} rows timed-out.  See {MISSING_TXT}")
    else:
        print("✅  All rows processed successfully.")

    print(f"CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
