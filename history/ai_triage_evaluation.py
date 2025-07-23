import difflib, os, re, requests, yaml

import pandas as pd

from box import Box
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from openai import OpenAI

from utils.logger import PRED_LOGGER, ERR_LOGGER, setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args


# llm_config = {
#     "config_list": [
#         {
#             "model":    "deepseek-ai/DeepSeek-V3",
#             "base_url": "https://api.siliconflow.cn/v1",
#             "api_key":  "sk-zlhpbuynjtlhysdqbqwfcuglzwdxxfhlasuamnsmkselhqto",
#         }
#     ]
# }

# def find_similar_specialties(specialty_list, text, cutoff=0.6):
#     text_lower = text.lower()
#     found = []
    
#     for specialty in specialty_list:
#         specialty_lower = specialty.lower()
        
#         # Exact match first
#         if specialty_lower in text_lower:
#             found.append(specialty)
#             continue

#         # Approximate match: compare specialty against text chunks (sliding window)
#         text_words = text_lower.split()
#         spec_words = specialty_lower.split()
#         window_size = len(spec_words)

#         # Sliding window over the text
#         for i in range(len(text_words) - window_size + 1):
#             text_chunk = ' '.join(text_words[i:i+window_size])
#             similarity = difflib.SequenceMatcher(None, specialty_lower, text_chunk).ratio()
#             if similarity >= cutoff:
#                 found.append(specialty)
#                 break

#     if not found:
#         return None
#     return list(set(found))

DEBUG = False
INPUT_DATA_PATH   = "datasets/mtsamples_layman_v5.csv"
API_KEY_DEEPSEEK  = "sk-zlhpbuynjtlhysdqbqwfcuglzwdxxfhlasuamnsmkselhqto"
BASE_URL_DEEPSEEK = "https://api.siliconflow.cn/v1"
API_KEY_UF        = "sk-JoOzBJv4uLYVjOKFU5if1w"
BASE_URL_UF       = "https://api.ai.it.ufl.edu/v1"


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).lower().strip()

def find_similar_specialties(specialty_list, text, cutoff=0.75):
    normalized_text = normalize_text(text)
    normalized_specialties = [normalize_text(s) for s in specialty_list]

    # First try exact match
    for original, normalized in zip(specialty_list, normalized_specialties):
        if normalized == normalized_text:
            return [original]

    # Then try fuzzy matching
    close_matches = difflib.get_close_matches(normalized_text, normalized_specialties, n=1, cutoff=cutoff)
    if close_matches:
        matched_index = normalized_specialties.index(close_matches[0])
        return [specialty_list[matched_index]]

    return None


def extract_specialty_from_response(response_text):
    # Try strict extraction first
    match = re.search(r'Specialty:\s*(.+)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to guess from any line that looks like a specialty
    lines = response_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith("patient") and not line.lower().startswith("task"):
            return line
    return response_text.strip()




def remove_think_tags(text):
    # Use re.DOTALL to match across newlines
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()  # Remove any leading/trailing whitespace



def predict_specialty_whole_description(description, target_llm, specialty_str, specialty_list,
                                        specialty_similarity, api_key, base_url):
    # Build prompt
    # prompt = f"\"{description}\"\nBased on the aforementioned description, choose the correct medical " \
    #          f"specialty from the following list without any other response: {specialty_str}"

    sys_prompt = f"""
You are a helpful medical assistant who helps assign medical specialty for patients.
Task: Based **only** on the Patient Description, pick the most appropriate medical specialty for this patient **from the following list**:
{specialty_str}

Respond ONLY in this format:
Specialty: <chosen_specialty>
"""

    # Choose backend
 
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    try:
        response = client.chat.completions.create(
            model=target_llm,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": description}
            ],
            timeout=60,  # Uncomment if your SDK version supports it
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        response_text = "api-error"


    # spec_match = re.search(r'Specialty:\s*([^\n\r]+)', response_text, re.IGNORECASE)
    # spec_str = spec_match.group(1).strip() if spec_match else response_text.strip()

    spec_str = extract_specialty_from_response(response_text)

    ans = find_similar_specialties(specialty_list, spec_str, specialty_similarity)
    predicted_specialty = "none" if ans is None or len(ans) > 1 else ans[0]

    return predicted_specialty, response_text



def main(opt):
    data = pd.read_csv(opt.data_root)
    # data = data.iloc[:2]

    # Evaluate LLM predictions
    ground_truths = []
    predictions = []
    row_ids = []
    layman_desc = []
    orig_desc = []
    target_llm = opt.model_name
    # List of medical specialties (same for all samples)
    specialty_str = f"'Cardiovascular / Pulmonary', 'Orthopedic', 'Obstetrics / Gynecology', "        \
                    f"'Sleep Medicine', 'Podiatry', "        \
                    f"'Gastroenterology', 'Neurology / Neurosurgery', "   \
                    f"'Urology', "       \
                    f"'ENT - Otolaryngology', 'Ophthalmology', 'Psychiatry / Psychology', " \
                    f"'Dermatology'"
    
    specialty_list = [
        # 'Allergy / Immunology',
        'Cardiovascular / Pulmonary',
        'Orthopedic',
        # 'Radiology',
        'Urology',
        'ENT - Otolaryngology',
        'Ophthalmology',
        'Psychiatry / Psychology',
        'Dermatology',
        # 'General Medicine',
        'Sleep Medicine',
        # 'Rheumatology',
        # 'Nephrology',
        # 'Hematology - Oncology',
        'Gastroenterology',
        # 'Endocrinology',
        'Obstetrics / Gynecology', 'Neurology / Neurosurgery', 'Podiatry'
    ]

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="    Processing"):
        desc = row['description']
        if row['medical_specialty'] != "Obstetrics / Gynecology": continue #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        actual_specialty = row['medical_specialty'].strip()

        # LLM prediction
        # pred, response = predict_specialty_whole_description(desc, target_llm, 
        #                                                      specialty_str, specialty_list, 
        #                                                      opt.specialty_similarity)
        pred, response = predict_specialty_whole_description(
                            desc,
                            opt.model_name,
                            specialty_str,
                            specialty_list,
                            opt.specialty_similarity,
                            opt.api_key, opt.base_url
                        )

        row_ids.append(row['row_id'])
        layman_desc.append(row['description'])
        orig_desc.append(row['original_description'])
        # Record results
        ground_truths.append(actual_specialty.strip().lower())
        predictions.append(pred.strip().lower())


        PRED_LOGGER.info(f"row_id {row['row_id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred}\n")
        
        if pred.strip().lower() != actual_specialty.strip().lower():
            ERR_LOGGER.info(f"row_id {row['row_id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | {response}\n")
            
        if DEBUG:
            break #!!!!!!!!!!!!

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    PRED_LOGGER.info(f"\n\n✅ Model Accuracy: {accuracy:.2%}")

    # Save detailed results
    results_df = pd.DataFrame({
        'row_id': row_ids,
        'description': layman_desc,
        'actual_specialty': ground_truths,
        'predicted_specialty': predictions,
        'original_description': orig_desc
    })

    results_df.to_csv(os.path.join(opt.save_dir, 
                                   f"triage_pred_{opt.basename}.csv"), 
                      index=False)

    print("Finished.")


if __name__ == "__main__":

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    llm_models = [
                #   "deepseek-ai/DeepSeek-V3",
                  "gpt-4.1",
                #   "claude-4-sonnet",
                #   "gemini-2.0-flash",
                #   "llama-3.3-70b-instruct",
                  ]

    if DEBUG:
        print("\n⚠️ ------------ ")
        print("DEBUG mode\n")

    for model_name in tqdm(llm_models, total=len(llm_models), desc="LLM MODELS"):
        opt = Box()
        opt.cur_time = get_cur_time()
        opt.model_name = model_name
        opt.data_root = INPUT_DATA_PATH
        opt.basename = os.path.basename(opt.model_name)

        print(f"\nUsing {model_name} ...\n")

        opt.save_dir    = os.path.join("res", f"{opt.cur_time}_triage_{opt.basename}")

        opt.codes_dir   = os.path.join(opt.save_dir, "codes")
        opt.log_dir     = os.path.join(opt.save_dir, 'log')

        create_folders(opt.save_dir)
        source_paths = ['utils/', 'ai_triage_evaluation.py']
        save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)
        setup_logger(opt.log_dir)

        if model_name == "deepseek-ai/DeepSeek-V3":
            opt.api_key  = API_KEY_DEEPSEEK
            opt.base_url = BASE_URL_DEEPSEEK
        else:
            opt.api_key  = API_KEY_UF
            opt.base_url = BASE_URL_UF

        opt.specialty_similarity = 0.8

        main(opt)
        if DEBUG: break #!
