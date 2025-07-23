API_KEY_DEEPSEEK  = "sk-zlhpbuynjtlhysdqbqwfcuglzwdxxfhlasuamnsmkselhqto"
BASE_URL_DEEPSEEK = "https://api.siliconflow.cn/v1"
API_KEY_UF        = "sk-JoOzBJv4uLYVjOKFU5if1w"
BASE_URL_UF       = "https://api.ai.it.ufl.edu/v1"

LLM_EXPERTS = [
    {
        "model_name": "llama-3.3-70b-instruct",
        "api_key": API_KEY_UF, "base_url": BASE_URL_UF
    },
    {
        "model_name": "gpt-4.1",
        "api_key": API_KEY_UF, "base_url": BASE_URL_UF
    },
    {
        "model_name": "claude-4-sonnet",
        "api_key": API_KEY_UF, "base_url": BASE_URL_UF
    },
    {
        "model_name": "gemini-2.0-flash",
        "api_key": API_KEY_UF, "base_url": BASE_URL_UF
    },
]

LLM_SGNN = {
    "model_name": "gpt-4o",
    "api_key": API_KEY_UF, 
    "base_url": BASE_URL_UF
}