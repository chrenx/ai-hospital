import csv, json, os, re

from box import Box
from openai import OpenAI
from tqdm import tqdm

from utils.logger import PRED_LOGGER, ERR_LOGGER, setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args


DEBUG             = False
INPUT_JSON        = "datasets/patients.json"
API_KEY_DEEPSEEK  = 
BASE_URL_DEEPSEEK = "https://api.siliconflow.cn/v1"
API_KEY_UF        = 
BASE_URL_UF       = "https://api.ai.it.ufl.edu/v1"
TRANSLATION_DIR   = "res/translation-cache"

SYS_PROMPT_TRANSLATOR = f"""
You are a professional medical translator. Please translate the following “medical_record” dictionary from Chinese to English, preserving all section headings and the meaning of each item, but do not summarize or add explanations. Output each key and value as plain English text. Keep the section order as in the original.
"""

SYS_PROMPT_REPORT = f"""
You are a medical assistant. Given the following “medical_record” (in Chinese), translate all information into clear, patient-friendly English for the patient and their family.
- Summarize each section (“Chief Complaint”, “History of Present Illness”, etc.) in easy-to-understand English.
- Explain any important medical terms (such as “stroke”, “hypertension”, etc.) in plain language.
- Emphasize the main diseases and their impact on health, treatment, and what to expect.
- Organize your summary using these headings:
1. Patient Information
2. Chief Complaint
3. History of Present Illness
4. Past Medical History
5. Physical Examination & Investigations
6. Diagnosis
7. Treatment
8. Recommendations & Follow-up
- If a section is missing, skip it.
- The output must be in English.

Please write the report in Markdown format. Use headings, bullet points, and bold for important keywords.

"""

# def medical_record_to_text(record):
#     out = []
#     for key, value in record.items():
#         # Remove leading/trailing whitespace from keys and values
#         section = f"{key.strip()}:\n{value.strip()}\n"
#         out.append(section)
#     return "\n".join(out)

def medical_record_to_text(medical_record):
    """Turn dict into readable text (Chinese keys preserved, order not guaranteed)."""
    lines = []
    for k, v in medical_record.items():
        v = v.strip() if isinstance(v, str) else str(v)
        lines.append(f"[{k}]\n{v}\n")
    return "\n".join(lines)


def build_prompt(medical_record):
    prompt = f"""
Here is the “medical_record” content (in Chinese):
{medical_record_to_text(medical_record)}
"""
    # {json.dumps(medical_record, ensure_ascii=False, indent=2)}
    return prompt


def call_llm(api_key, base_url, model_name, prompt, sys_prompt):
    """
    Replace this with your LLM call (e.g., OpenAI, Qwen, etc.)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model    = model_name,
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            timeout=60,  # Uncomment if your SDK version supports it
        )
        response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        response = "api-error"
    return response


def strip_markdown_fence(md_str):
    """
    Remove leading/trailing triple backticks (with or without 'markdown') from LLM output.
    """
    # Remove leading fence
    md_str = re.sub(r"^\s*```(?:markdown)?\s*\n?", "", md_str, flags=re.IGNORECASE)
    # Remove trailing fence
    md_str = re.sub(r"\n?```\s*$", "", md_str)
    return md_str.strip()


def translate_medical_record(case_id, medrec_text_zh, call_llm_func):
    path = os.path.join(TRANSLATION_DIR, f"{case_id}_medical_record_en.txt")
    if os.path.exists(path):
        # Already translated, just read from file
        with open(path, "r", encoding="utf-8") as f:
            medrec_text_en = f.read()
    else:
        # Call LLM to translate, then save
        medrec_text_en = call_llm_func(API_KEY_DEEPSEEK, BASE_URL_DEEPSEEK, 
                                  "deepseek-ai/DeepSeek-V3", medrec_text_zh, SYS_PROMPT_TRANSLATOR)
        with open(path, "w", encoding="utf-8") as f:
            f.write(medrec_text_en)
    return medrec_text_en


def main(opt):
    with open(opt.data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    error_cases = []

    for case in tqdm(cases, total=len(cases), desc="    Processing"):
        case_id = case.get("id", "unknown")
        medrec = case.get("medical_record", {})

        # --- Generate user-friendly report via LLM ---
        prompt = build_prompt(medrec)
        user_report = call_llm(opt.api_key, opt.base_url, opt.model_name, prompt, SYS_PROMPT_REPORT)
        
        if user_report == "api-error":
            error_cases.append({
                "id": case_id,
                "user_report_error": user_report == "api-error",
                "medrec_text_en_error": medrec_text_en == "api-error"
            })
            print(f"API error for case {case_id}")
            continue

        user_report = strip_markdown_fence(user_report)

        # --- Convert medical_record to text ---
        medrec_text_zh = medical_record_to_text(medrec)
        medrec_text_en = translate_medical_record(case_id, medrec_text_zh, call_llm)

        # --- Save outputs ---
        out_dir = os.path.join(opt.save_dir, "medical-cases", str(case_id))
        os.makedirs(out_dir, exist_ok=True)


        with open(os.path.join(out_dir, "user_friendly_report_en.md"), "w", encoding="utf-8") as f:
            f.write(user_report)
        with open(os.path.join(out_dir, "orig_medical_record_zh.txt"), "w", encoding="utf-8") as f:
            f.write(medrec_text_zh.strip() + "\n")
        with open(os.path.join(out_dir, "orig_medical_record_en.txt"), "w", encoding="utf-8") as f:
            f.write(medrec_text_en.strip() + "\n")

        if DEBUG: break #!

    if error_cases:
        with open(os.path.join(opt.save_dir, "api_error_cases.csv"), "w", 
                  encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "user_report_error", "medrec_text_en_error"])
            writer.writeheader()
            writer.writerows(error_cases)
        print(f"Wrote {len(error_cases)} error cases to api_error_cases.csv")
    print("DONE ✅.")


if __name__ == "__main__":
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    llm_models = ["deepseek-ai/DeepSeek-V3",
                  "gpt-4.1",
                  "claude-4-sonnet",
                  "gemini-1.5-pro",
                  "gemma-3-27b-it",
                  "llama-3.3-70b-instruct",
                  "mixtral-8x7b-instruct",
                  "codestral-22b",
                  ]
    
    if DEBUG:
        print("\n⚠️ ------------ ")
        print("DEBUG mode\n")

    for model_name in tqdm(llm_models, total=len(llm_models), desc="LLM MODELS"):
        opt = Box()
        opt.cur_time = get_cur_time()
        opt.model_name = model_name
        opt.data_path = INPUT_JSON

        print(f"\nUsing {model_name} ...\n")

        opt.save_dir    = os.path.join("res", 
                                    f"{opt.cur_time}_discharge_{os.path.basename(opt.model_name)}")

        opt.codes_dir   = os.path.join(opt.save_dir, "codes")
        opt.log_dir     = os.path.join(opt.save_dir, 'log')

        create_folders(opt.save_dir)
        source_paths = ['utils/', 'ai_discharge.py']
        save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)

        if model_name == "deepseek-ai/DeepSeek-V3":
            opt.api_key  = API_KEY_DEEPSEEK
            opt.base_url = BASE_URL_DEEPSEEK
        else:
            opt.api_key  = API_KEY_UF
            opt.base_url = BASE_URL_UF
        main(opt)
        if DEBUG: break #!