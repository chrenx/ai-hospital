import difflib, os, re, yaml

import pandas as pd
import ollama

from box import Box
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils.logger import PRED_LOGGER, ERR_LOGGER, setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args



def find_similar_specialties(specialty_list, text, cutoff=0.6):
    text_lower = text.lower()
    found = []
    
    for specialty in specialty_list:
        specialty_lower = specialty.lower()
        
        # Exact match first
        if specialty_lower in text_lower:
            found.append(specialty)
            continue

        # Approximate match: compare specialty against text chunks (sliding window)
        text_words = text_lower.split()
        spec_words = specialty_lower.split()
        window_size = len(spec_words)

        # Sliding window over the text
        for i in range(len(text_words) - window_size + 1):
            text_chunk = ' '.join(text_words[i:i+window_size])
            similarity = difflib.SequenceMatcher(None, specialty_lower, text_chunk).ratio()
            if similarity >= cutoff:
                found.append(specialty)
                break

    if not found:
        return None
    return list(set(found))


def remove_think_tags(text):
    # Use re.DOTALL to match across newlines
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()  # Remove any leading/trailing whitespace


# Function to query Ollama's LLM to classify the specialty based on whole description from patients
def predict_specialty_whole_description(description, target_llm, specialty_str, specialty_list,
                                        specialty_similarity):
    # Build the system message
    
    content = f"\"{description}\"\n Based on the aforementioned description, "\
        f"choose the correct medical "\
        f"specialty from the following list without any other response: {specialty_str}"

    response = ollama.generate(model=target_llm, prompt=content)
    response = response.response.strip()

    # remove <think> ... </think> for deepseek r1
    if target_llm == "deepseek-r1:14b":
        response = remove_think_tags(response)

    ans = find_similar_specialties(specialty_list, response, specialty_similarity)
    if ans == None or len(ans) > 1:
        predicted_specialty = "none"
    else:
        predicted_specialty = ans[0]

    return predicted_specialty, response


def main(opt):
    data = pd.read_csv(opt.data_root)
    # data = data.iloc[:2]

    # Evaluate LLM predictions
    ground_truths = []
    predictions = []
    target_llm = opt.model_name
    # List of medical specialties (same for all samples)
    specialty_str = f"'Allergy / Immunology', 'Cardiovascular / Pulmonary', "        \
                    f"'General Medicine', 'Sleep Medicine', 'Rheumatology', "        \
                    f"'Nephrology', 'Hematology - Oncology', 'Gastroenterology', "   \
                    f"'Endocrinology'"
    
    specialty_list = [
            'Allergy / Immunology',
            'Cardiovascular / Pulmonary',
            'General Medicine',
            'Sleep Medicine',
            'Rheumatology',
            'Nephrology',
            'Hematology - Oncology',
            'Gastroenterology',
            'Endocrinology'
        ]

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        desc = row['description']
        actual_specialty = row['medical_specialty'].strip()

        # LLM prediction
        pred, response = predict_specialty_whole_description(desc, target_llm, 
                                                             specialty_str, specialty_list, 
                                                             opt.specialty_similarity)
        

        # Record results
        ground_truths.append(actual_specialty.strip().lower())
        predictions.append(pred.strip().lower())


        PRED_LOGGER.info(f"row_id {row['row_id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred}\n")
        
        if pred.strip().lower() != actual_specialty.strip().lower():
            ERR_LOGGER.info(f"row_id {row['row_id']} - Actual: {actual_specialty} "
                         f"| Predicted: {pred} | {response}\n")
        

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    PRED_LOGGER.info(f"\n\n✅ Model Accuracy: {accuracy:.2%}")

    # Save detailed results
    results_df = pd.DataFrame({
        'row_id': data['row_id'],
        'description': data['description'],
        'actual_specialty': ground_truths,
        'predicted_specialty': predictions
    })
    results_df.to_csv(os.path.join(opt.save_dir, 
                                   f"triage_pred_{os.path.basename(opt.model_name)}.csv"), 
                      index=False)

    print("Finished.")


if __name__ == "__main__":
    config_path = "ai_triage_evaluation_opt.yaml"
    with open(config_path, "r") as f:
        opt = Box(yaml.safe_load(f))
    
    opt.cur_time    = get_cur_time()
    opt.save_dir    = os.path.join(opt.save_dir, 
                                   f"{opt.cur_time}_{os.path.basename(opt.model_name)}")
    opt.codes_dir   = os.path.join(opt.save_dir, "codes")
    opt.log_dir     = os.path.join(opt.save_dir, 'log')

    create_folders(opt.save_dir)
    source_paths = ['utils/', 'ai_triage_evaluation.py', config_path]
    save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)
    setup_logger(opt.log_dir)

    main(opt)
