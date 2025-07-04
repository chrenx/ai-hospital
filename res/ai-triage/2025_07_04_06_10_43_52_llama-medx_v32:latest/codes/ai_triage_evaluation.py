import difflib, os, re, requests, yaml

import pandas as pd
import ollama

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



def predict_specialty_whole_description(description, target_llm, specialty_str, specialty_list,
                                        specialty_similarity, llm_backend="ollama", api_config=None):
    # Build prompt
    # prompt = f"\"{description}\"\nBased on the aforementioned description, choose the correct medical " \
    #          f"specialty from the following list without any other response: {specialty_str}"

    prompt = f"""Patient Description:
        \"{description}\"

        Task: Based **only** on the above, pick the most appropriate medical specialty for this patient **from the following list**:
        {specialty_str}

        Respond ONLY in this format:
        Specialty: <chosen_specialty>
        """

    # Choose backend
    if llm_backend == "ollama":
        import ollama
        response_text = ollama.generate(model=target_llm, prompt=prompt).response.strip()

    elif llm_backend == "api":
        assert api_config is not None, "api_config must be provided when llm_backend='api'"
        client = OpenAI(
            api_key=api_config["api_key"],
            base_url=api_config["base_url"]
        )
        try:
            response = client.chat.completions.create(
                model=api_config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                timeout=60,  # Uncomment if your SDK version supports it
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            response_text = "api-error"
    else:
        raise ValueError(f"Unsupported backend: {llm_backend}")

    if target_llm == "deepseek-r1:14b":
        response_text = remove_think_tags(response_text)

    spec_match = re.search(r'Specialty:\s*([^\n\r]+)', response_text, re.IGNORECASE)
    spec_str = spec_match.group(1).strip() if spec_match else response_text.strip()

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
    specialty_str = f"'Allergy / Immunology', 'Cardiovascular / Pulmonary', "        \
                    f"'Sleep Medicine', 'Rheumatology', "        \
                    f"'Nephrology', 'Hematology - Oncology', 'Gastroenterology', "   \
                    f"'Endocrinology', 'Orthopedic', 'Radiology', 'Urology', "       \
                    f"'ENT - Otolaryngology', 'Ophthalmology', 'Psychiatry / Psychology', " \
                    f"'Dermatology'"
    
    specialty_list = [
        'Allergy / Immunology',
        'Cardiovascular / Pulmonary',
        'Orthopedic',
        # 'Radiology',
        'Urology',
        'ENT - Otolaryngology',
        'Ophthalmology',
        'Psychiatry / Psychology',
        'Dermatology',
        'Sleep Medicine',
        'Rheumatology',
        # 'Nephrology',
        # 'Hematology - Oncology',
        'Gastroenterology',
        'Endocrinology'
    ]

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        desc = row['description']
        actual_specialty = row['medical_specialty'].strip()

        # LLM prediction
        # pred, response = predict_specialty_whole_description(desc, target_llm, 
        #                                                      specialty_str, specialty_list, 
        #                                                      opt.specialty_similarity)
        pred, response = predict_specialty_whole_description(
                            desc,
                            target_llm,
                            specialty_str,
                            specialty_list,
                            opt.specialty_similarity,
                            llm_backend=opt.llm_backend,
                            api_config=opt.api_config if opt.llm_backend == "api" else None
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
            
        if opt.debug:
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
    config_path = "ai_triage_evaluation_opt.yaml"
    with open(config_path, "r") as f:
        opt = Box(yaml.safe_load(f))
    
    # if opt.llm_backend == "api":
    #     basename = os.path.basename(opt.api_config.model)
    # else:
    #     basename = os.path.basename(opt.model_name)

    opt.debug = False

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    llm_models = ["gpt-4.1",
                  "claude-4-sonnet",
                  "claude-4-opus",
                  "gemini-1.5-pro",
                  "gemma-3-27b-it",
                  "granite-3.1-8b-instruct", 
                  "llama-3.3-70b-instruct",
                  "mixtral-8x7b-instruct",
                  "codestral-22b",
                  ]
    opt.llm_backend = "api"
    for model_name in llm_models:
        opt.model_name = model_name
        opt.cur_time    = get_cur_time()

        opt.basename = os.path.basename(model_name)
        print(f"\nUsing {opt.basename} ...\n")

        opt.save_dir    = os.path.join("res", 
                                    f"{opt.cur_time}_{opt.basename}")

        opt.codes_dir   = os.path.join(opt.save_dir, "codes")
        opt.log_dir     = os.path.join(opt.save_dir, 'log')

        create_folders(opt.save_dir)
        source_paths = ['utils/', 'ai_triage_evaluation.py', config_path]
        save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)
        setup_logger(opt.log_dir)

        main(opt)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    opt.llm_backend = "ollama"
    model_names = ["richardyoung/llama-medx_v32:latest",
                    "koesn/llama3-openbiollm-8b:q6_K", 
                    "richardyoung/llama-medx_v32:latest", 
                    "ahmgam/medllama3-v20:latest",
                    "meditron:7b",
                    "deepseek-v2:16b",
                    "deepseek-r1:14b",
                    ]
    
    for model_name in model_names:
        opt.model_name = model_name
        opt.cur_time    = get_cur_time()

        opt.basename = os.path.basename(model_name)
        print(f"\nUsing {opt.basename} ...\n")

        opt.save_dir    = os.path.join("res", 
                                    f"{opt.cur_time}_{opt.basename}")

        opt.codes_dir   = os.path.join(opt.save_dir, "codes")
        opt.log_dir     = os.path.join(opt.save_dir, 'log')

        create_folders(opt.save_dir)
        source_paths = ['utils/', 'ai_triage_evaluation.py', config_path]
        save_codes_args(source_paths, opt, opt.save_dir, opt.codes_dir)
        setup_logger(opt.log_dir)
        main(opt)

        

    
