import difflib, os, re, requests, yaml, json
import logging
import pandas as pd

from box import Box
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from openai import OpenAI

from utils.logger import setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args
import prompts
from yes_no_checker import ask_yes_no_specialty_check, TriageCorrector, DialogueTriageAssistant



DEBUG = False
INPUT_DATA_PATH   = "datasets/new_patients.json"
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


def extract_confidence_from_response(response_text):
    match = re.search(r'Confidence\s*:\s*(\d+)', response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1  # 表示未提取到


def extract_specialty_from_response(response_text, specialty_list):
    # First, try "Specialty: ..." match
    match = re.search(r'Specialty:\s*([^\n\r:，。]+)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Check for exact specialty mentions (after lowercasing both sides)
    hits = [s for s in specialty_list if s in response_text]
    if len(hits) == 1:
        return hits[0]  # Only one specialty appeared — use it
    
    # Fallback: find first valid-looking Chinese line
    lines = response_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("patient") or line.lower().startswith("task"):
            continue
        # Try to extract only Chinese content or substring after colon
        if ":" in line:
            parts = line.split(":")
            candidate = parts[-1].strip()
            if re.search(r'[\u4e00-\u9fff]', candidate):
                return candidate
        elif re.search(r'[\u4e00-\u9fff]', line):
            return line
    return response_text.strip()


def remove_think_tags(text):
    # Use re.DOTALL to match across newlines
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()  # Remove any leading/trailing whitespace



def predict_specialty_whole_description(description, target_llm, specialty_str, specialty_list,
                                        specialty_similarity, api_key, base_url, opt, history=""):
    # Build prompt
    # prompt = f"\"{description}\"\nBased on the aforementioned description, choose the correct medical " \
    #          f"specialty from the following list without any other response: {specialty_str}"



    sys_prompt = prompts.TRIAGE
 
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

    spec_str = extract_specialty_from_response(response_text, specialty_list)
    confidence_score = extract_confidence_from_response(response_text)

    ans = find_similar_specialties(specialty_list, spec_str, specialty_similarity)

    predicted_specialty = "none" if ans is None or len(ans) > 1 else ans[0]


    if predicted_specialty == "none" or confidence_score < 90:
        # print("confidence:", confidence_score)
        dialogue_assistant = DialogueTriageAssistant(
            model_name=opt.model_name,
            api_key=opt.api_key,
            base_url=opt.base_url,
            verbose=True,
            pred_logger=opt.pred_logger,
        )
        candidates = dialogue_assistant.run(description + f"\n既往史: \n{history}")
        if len(candidates) == 1:
            predicted_specialty = candidates[0]
            response_text += f"\n<对话式纠正> -> {predicted_specialty}\n"
        elif len(candidates) > 1:
            predicted_specialty = "none"  # 或 fallback to main model
            response_text += f"\n<对话式辅助> 剩余候选：{','.join(candidates)}\n"

    return predicted_specialty, response_text



def main(opt):
    with open(opt.data_root, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = data.iloc[:2]

    # Evaluate LLM predictions
    ground_truths = []
    predictions = []
    row_ids = []
    layman_desc = []
    all_response = []
    orig_desc = []
    target_llm = opt.model_name
    # List of medical specialties (same for all samples)
    specialty_str = f"儿科病例, 耳鼻咽喉科病例, 妇产科病例, 外科病例, 内科病例, 眼科病例, 口腔科病例"
    
    specialty_list = ["儿科病例", "耳鼻咽喉科病例", "妇产科病例", "外科病例", "内科病例", "眼科病例", "口腔科病例"]

    for entry in tqdm(data, total=len(data), desc="    Processing"):
        gen_info = entry["medical_record"].get("一般资料", "")
        # complaint = entry["medical_record"].get("主诉", "")
        complaint = re.search(r"<病情陈述>(.*?)<性别>", entry["profile"], re.DOTALL).group(1).strip()

        desc = f"{gen_info.strip()}\n{complaint.strip()}"
        history = entry["medical_record"].get("现病史", "")

        # actual_specialty = entry.get("specialty", "").strip()
        actual_specialty = (entry.get("specialty") or "").strip()

        if actual_specialty == "":
            opt.err_logger.info(f"Not Valid: ID {entry['id']}, specialty: {entry['department']}, {complaint}\n")
            continue


        pred, response = predict_specialty_whole_description(
                            desc,
                            opt.model_name,
                            specialty_str,
                            specialty_list,
                            opt.specialty_similarity,
                            opt.api_key, opt.base_url,
                            opt=opt,
                            history=history,
                        )

        row_ids.append(entry['id'])
        layman_desc.append(desc)
        # Record results
        ground_truths.append(actual_specialty.strip())
        predictions.append(pred.strip())
        all_response.append(response)


        opt.pred_logger.info(f"row_id {entry['id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | Response: {response}\n")
        
        if pred.strip().lower() != actual_specialty.strip().lower():
            opt.err_logger.info(f"row_id {entry['id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | Response: {response}\n")
            
        if DEBUG:
            break #!!!!!!!!!!!!

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    opt.pred_logger.info(f"\n\n✅ Model Accuracy: {accuracy:.2%}")

    # Save detailed results
    results_df = pd.DataFrame({
        'row_id': row_ids,
        'actual_specialty': ground_truths,
        'predicted_specialty': predictions,
        'response_text': all_response, 
        'description': layman_desc,
    })

    results_df.to_csv(os.path.join(opt.save_dir, 
                                   f"triage_pred_{opt.basename}.csv"), 
                                    index=False)

    print("Finished.")


if __name__ == "__main__":

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    llm_models = [
                  "llama-3.3-70b-instruct",
                #   "gpt-4.1",
                #   "claude-4-sonnet",
                #   "gemini-2.0-flash",
                #   "deepseek-ai/DeepSeek-V3"
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

        opt.pred_logger = logging.getLogger(f"pred_log_{opt.basename}")
        opt.err_logger = logging.getLogger(f"error_log_{opt.basename}")

        setup_logger(opt.log_dir, opt.pred_logger, opt.err_logger)

        if model_name == "deepseek-ai/DeepSeek-V3":
            opt.api_key  = API_KEY_DEEPSEEK
            opt.base_url = BASE_URL_DEEPSEEK
        else:
            opt.api_key  = API_KEY_UF
            opt.base_url = BASE_URL_UF

        opt.specialty_similarity = 0.8

        main(opt)
        if DEBUG: break #!
