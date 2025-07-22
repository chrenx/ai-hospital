
from openai import OpenAI
import re

SPECIALTY_QUESTIONS = [
    ("口腔科病例", "患者描述是否涉及牙齿、牙龈、口腔不适？"),
    ("耳鼻咽喉科病例", "是否描述了耳痛、鼻塞、咽痛或声音嘶哑等症状？"),
    ("眼科病例", "是否出现视力下降、眼睛红、异物感等问题？"),
    ("妇产科病例", "是否描述了女性生殖系统问题或孕产史？"),
    # ("儿科病例", "是否明确是儿童或婴幼儿？"),
    ("外科病例", "是否存在外伤、疼痛、肿块、麻木或手术史？"),
    ("内科病例", "是否有发烧、咳嗽、消化、内分泌等系统症状？")
]

def build_yn_prompt(question: str) -> str:
    return f"""你是医院的智能辅助分诊系统。请根据病人的主观描述回答以下问题：

问题：{question}

你只能回答：“是”、“否”或“不确定”。请不要添加解释说明。
"""

def normalize_response(text):
    if "是" in text:
        return "是"
    elif "否" in text:
        return "否"
    elif "不确定" in text:
        return "不确定"
    return "无法解析"



class DialogueTriageAssistant:
    def __init__(self, model_name, api_key, base_url, verbose=False, pred_logger=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.verbose = verbose
        self.pred_logger = pred_logger

    def ask_question(self, profile, question):
        prompt = build_yn_prompt(question)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": profile}
                ],
                timeout=30
            )
            answer = response.choices[0].message.content.strip()
            norm_answer = normalize_response(answer)
            return norm_answer
        except Exception as e:
            print(f"[DialogueTriageAssistant ERROR] {e}")
            return "无法解析"

    def run(self, profile):
        candidate_specialties = set([q[0] for q in SPECIALTY_QUESTIONS])

        for specialty, question in SPECIALTY_QUESTIONS:
            answer = self.ask_question(profile, question)
            if self.verbose:
                self.pred_logger.info(f"[{specialty}] Q: {question} → A: {answer}\n")

            if answer == "否":
                candidate_specialties.discard(specialty)

        return list(candidate_specialties)




def ask_yes_no_specialty_check(description, specialty_name, api_key, base_url, target_llm="llama-3.3-70b-instruct"):
    """
    Ask LLM: 是否该描述可能属于指定 specialty。
    Returns: "是" / "否" / "不确定"
    """

    sys_prompt = f"""你是医院智能分诊系统中的辅助决策模块。你现在要判断一个病人的主观描述是否可能属于指定的科室。

请严格按以下格式输出回答：
回答（只能选“是”或“否”或“不确定”）

判断原则：
1. 不要依赖年龄判断；
2. 请只依据是否提到了与“{specialty_name}”相关的典型症状或部位；
3. 若信息不足以判断，请答“不确定”。

只输出“是”、“否”或“不确定”，不要多余解释。
"""

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=target_llm,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": description}
            ],
            timeout=30
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Yes/No API ERROR] {e}")
        return "API错误"

    if "是" in text:
        return "是"
    elif "否" in text:
        return "否"
    elif "不确定" in text:
        return "不确定"
    else:
        return "无法解析"






class TriageCorrector:
    def __init__(self, target_specialties, model_name, api_key, base_url, verbose=False, pred_logger=None):
        self.specialties = target_specialties  # e.g., ["口腔科病例", "耳鼻咽喉科病例"]
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.verbose = verbose
        self.pred_logger = pred_logger

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _build_prompt(self, specialty):
        return f"""你是医院智能分诊系统中的辅助判断模块。你需要判断一个病人的主观描述是否可能属于以下科室之一：
"{specialty}"

判断依据：
- 仅根据描述中提到的症状和部位做判断；
- 不要使用医学术语或诊断信息；
- 不可因年龄直接判断为“儿科”；
- 如果信息不明确，请回答“不确定”。

你只能回答：“是”、“否” 或 “不确定”，不要添加解释。"""

    def ask_yes_no(self, description, specialty):
        prompt = self._build_prompt(specialty)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": description}
                ],
                timeout=30
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[TriageCorrector ERROR] {e}")
            return "api-error"

        return answer

    def should_redirect_to_specialty(self, description, specialty):
        answer = self.ask_yes_no(description, specialty)
        if self.verbose and self.pred_logger:
            self.pred_logger.info(f"     [Corrector] {specialty} -> {answer}")
        return answer == "是"

