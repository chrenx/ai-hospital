

from utils.logger import PRED_LOGGER, ERR_LOGGER, setup_logger
from utils.tools import create_folders, get_cur_time, save_codes_args


def main(opt):
    pass

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
    opt.llm_backend = "ollama"
    model_names = [
                    # "richardyoung/llama-medx_v32:latest",
                    # "koesn/llama3-openbiollm-8b:q6_K", 
                    # "richardyoung/llama-medx_v32:latest", 
                    # "ahmgam/medllama3-v20:latest",
                    "deepseek-v2:16b",
                    "deepseek-r1:14b",
                    "meditron:7b",
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

        

    
