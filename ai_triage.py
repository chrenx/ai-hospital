import difflib, os, re, requests, yaml, json
import logging
import pandas as pd

from box import Box
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from openai import OpenAI

from config import LLM_SGNN, LLM_EXPERTS
from pipeline.sgnn import SGNN
from utils.logger import setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args





DEBUG = False
INPUT_DATA_PATH   = "datasets/new_patients.json"


# llm_models = [
#             "llama-3.3-70b-instruct",
#             "gpt-4.1",
#             "claude-4-sonnet",
#             "gemini-2.0-flash",
#             "deepseek-ai/DeepSeek-V3"
#             ]


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).lower().strip()


def extract_confidence_from_response(response_text):
    match = re.search(r'Confidence\s*:\s*(\d+)', response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1  # 表示未提取到


def predict_specialty_whole_description(case, specialty_list, specialty_similarity, opt):
    pipeline = SGNN(llm_sgnn=LLM_SGNN, llm_experts=LLM_EXPERTS, 
                    pred_logger=opt.pred_logger, err_logger=opt.err_logger, 
                    cutoff=opt.specialty_similarity)
    pred, voting, sgnn = pipeline.predict(case, specialty_list, specialty_similarity)

    return pred, voting, sgnn



def main(opt):
    with open(opt.data_root, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = data.iloc[:2]

    # Evaluate LLM predictions
    ground_truths = []
    predictions = []
    row_ids = []
    layman_desc = []
    all_voting = []

    # List of medical specialties (same for all samples)
    
    specialty_list = ["儿科病例", "耳鼻咽喉科病例", "妇产科病例", "外科病例", "内科病例", "眼科病例", "口腔科病例"]

    for entry in tqdm(data, total=len(data), desc="    Processing"):
        gen_info = entry["medical_record"].get("一般资料", "")
        # complaint = entry["medical_record"].get("主诉", "")
        complaint = re.search(r"<病情陈述>(.*?)<性别>", entry["profile"], re.DOTALL).group(1).strip()

        whole_desc = f"{gen_info.strip()}\n{complaint.strip()}"

        case = {"id": entry["id"], "desc": gen_info.strip(), "demographics": complaint.strip()}

        # actual_specialty = entry.get("specialty", "").strip()
        actual_specialty = (entry.get("specialty") or "").strip()

        if actual_specialty == "":
            opt.err_logger.info(f"Not Valid: ID {entry['id']}, specialty: {entry['department']}, {complaint}\n")
            continue

        pred, voting, sgnn = predict_specialty_whole_description(
                            case,
                            specialty_list,
                            opt.specialty_similarity,
                            opt=opt,
                        )

        whole_desc += f"\n{sgnn}"
        row_ids.append(entry['id'])
        layman_desc.append(whole_desc)
        # Record results
        ground_truths.append(actual_specialty.strip())
        predictions.append(pred.strip())
        all_voting.append(voting)


        opt.pred_logger.info(f"row_id {entry['id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | Voting: {voting}\n")
        
        if pred.strip().lower() != actual_specialty.strip().lower():
            opt.err_logger.info(f"row_id {entry['id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | Voting: {voting}\n")
            
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
        'response_text': all_voting, 
        'description': layman_desc,
    })

    results_df.to_csv(os.path.join(opt.save_dir, 
                                   f"triage_pred.csv"), 
                                    index=False)

    print("Finished.")


if __name__ == "__main__":

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if DEBUG:
        print("\n⚠️ ------------ ")
        print("DEBUG mode\n")

    opt = Box()
    opt.cur_time = get_cur_time()
    opt.data_root = INPUT_DATA_PATH

    opt.save_dir    = os.path.join("res", f"{opt.cur_time}_triage")
    opt.codes_dir   = os.path.join(opt.save_dir, "codes")
    opt.log_dir     = os.path.join(opt.save_dir, 'log')
    
    opt.specialty_similarity = 0.8

    create_folders(opt.save_dir)
    source_paths = ['utils/', 'pipeline/', 'ai_triage.py', 'config.py']
    save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)

    opt.pred_logger = logging.getLogger(f"pred_log")
    opt.err_logger = logging.getLogger(f"error_log")

    setup_logger(opt.log_dir, opt.pred_logger, opt.err_logger)

    main(opt)
