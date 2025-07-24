import difflib, json, re
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from openai import OpenAI


def visualize_graph(G):
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'relation')

        plt.figure(figsize=(8,6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Latent Symptom Graph")
        plt.show()


def save_sgnn_with_id(data, case_id, output_path="sgnn.json"):
    """
    Extract JSON from LLM output and append it to a .json file with the given ID.
    """

    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ File exists but is not valid JSON. Reinitializing.")
                all_data = {}
    else:
        all_data = {}

    all_data[case_id] = data

    # Save the updated dictionary
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved case {case_id} to {output_path}")


def extract_json_block(text):
    try:
        # Greedily find the first { ... } block
        match = re.search(r'{[\s\S]*}', text)
        if not match:
            raise ValueError("No JSON object found")

        json_str = match.group()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        # Optional: use a tolerant parser like `json5` or fix malformed JSON manually here
        return None
    

def extract_json_array(text):
    # Remove code block markers
    text_clean = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Fallback: regex extract bracket content if needed
    if not text_clean.startswith("["):
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            text_clean = match.group()

    try:
        return json.loads(text_clean)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(text)
        print()
        return None


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


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).lower().strip()


def find_similar_specialties(specialty_list, text, cutoff=0.8):
    normalized_text = normalize_text(text)
    normalized_specialties = [normalize_text(s) for s in specialty_list]

    # First try exact match
    for original, normalized in zip(specialty_list, normalized_specialties):
        if normalized == normalized_text:
            return [original]

    # Then try fuzzy matching
    close_matches = difflib.get_close_matches(normalized_text, normalized_specialties, n=1, 
                                              cutoff=cutoff)
    if close_matches:
        matched_index = normalized_specialties.index(close_matches[0])
        return [specialty_list[matched_index]]

    return None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class SGNN():
    def __init__(self, llm_sgnn, llm_experts, pred_logger, err_logger, 
                 sgnn_path="sgnn.json", cutoff=0.8):
        self.llm_sgnn    = llm_sgnn     # {"model_name":, "api_key":, "base_url":}
        self.llm_experts = llm_experts  # [{"model_name":, "api_key":, "base_url":}, ...]
        self.pred_logger = pred_logger
        self.err_logger  = err_logger
        self.cutoff      = cutoff
        self.sgnn_path   = sgnn_path
        self.sgnn_data   = None
        if Path(self.sgnn_path).exists():
            with open(self.sgnn_path, "r", encoding="utf-8") as f:
                self.sgnn_data = json.load(f)


    def predict(self, case, specialty_list, specialty_similarity):
        # case {"id": , "desc": , "demographics": }
        id, desc, demo = case["id"], case["desc"], case["demographics"]
        if self.sgnn_data is not None and id is not None and f"{id}" in self.sgnn_data:
            symptom_graph = self.sgnn_data[f"{id}"]
        else:
            structured_info, err_flag = self._extract_structured_info(desc)
            if err_flag:
                return "none", structured_info, None
            symptom_graph, err_flag = self._infer_symptom_graph(structured_info)
            if err_flag:
                return "none", symptom_graph, None
            # for s, t, r in symptom_graph:
            #     print(f"{s} --[{r}]--> {t}")

            save_sgnn_with_id(symptom_graph, id)

        pred, voting = self._classify_specialty_from_graph(demo, desc, 
                                                             symptom_graph, specialty_list)
        return pred, symptom_graph, voting


    def _classify_specialty_from_graph(self, demo, desc, graph_triples, specialty_list):
        specialty_str = ", ".join(specialty_list)
        graph_description = "\n".join([
            f"- “{s}” 与 “{t}” 的关系是：{r}" for s, t, r in graph_triples
        ])

        prompt = self._build_prompt_classify(demo, desc, graph_description, specialty_str)
        
        voting = {ele: 0 for ele in specialty_list}

        for model in self.llm_experts:

            client = OpenAI(
                api_key  = model["api_key"],
                base_url = model["base_url"],
            )

            sys_prompt = "你是一个专业医学专业的分诊系统，需要根据症状图谱信息判断应该将患者分诊到哪个科室。"
            try:
                response = client.chat.completions.create(
                    model=model["model_name"],
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=30,  # Uncomment if your SDK version supports it
                    temperature=0.2,
                )
                response = response.choices[0].message.content.strip()
                response = extract_specialty_from_response(response, specialty_list)
                ans      = find_similar_specialties(specialty_list, response, self.cutoff)
                response = "none" if ans is None or len(ans) > 1 else ans[0]
                if response in voting:
                    voting[response] += 1

            except Exception as e:
                print(f"API Error: {e}")
                response = e

        final_res = max(voting, key=voting.get)

        return final_res, str(voting)


    def _infer_symptom_graph(self, structured_info):
        symptoms = structured_info["symptoms"]
        demographics = structured_info.get("demographics", {})

        # Create a clean symptom list for the prompt
        symptom_terms = [s.get("term", "") for s in symptoms if "term" in s]
        symptom_list = "\n".join(f"- {term}" for term in symptom_terms)

        prompt = self._build_prompt_symptom_graph(symptom_list, demographics)
    
        client = OpenAI(
            api_key  = self.llm_sgnn["api_key"],
            base_url = self.llm_sgnn["base_url"],
        )

        err_flag = False
        try:
            response = client.chat.completions.create(
                model=self.llm_sgnn["model_name"],
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗图谱推理系统"},
                    {"role": "user", "content": prompt}
                ],
                timeout=30,  # Uncomment if your SDK version supports it
                temperature=0.2,
            )
            response = response.choices[0].message.content.strip()
            response = extract_json_array(response)
        except Exception as e:
            print(f"API Error: {e}")
            response = e
            err_flag = True

        if response is None: 
            err_flag = True
        else:
            G = nx.DiGraph()
            for s, t, r in response:
                G.add_edge(s, t, relation=r)


        return response, err_flag


    def _extract_structured_info(self, description):
        prompt = self._build_prompt_extract_structured_info(description)

        client = OpenAI(
            api_key  = self.llm_sgnn["api_key"],
            base_url = self.llm_sgnn["base_url"],
        )
        err_flag = False
        try:
            response = client.chat.completions.create(
                model=self.llm_sgnn["model_name"],
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗文本分析助手"},
                    {"role": "user", "content": prompt}
                ],
                timeout=30,  # Uncomment if your SDK version supports it
                temperature=0.2,
            )
            response = response.choices[0].message.content.strip()
            response = extract_json_block(response) # dict
        except Exception as e:
            print(f"API Error: {e}")
            response = e
            err_flag = True

        if response is None: err_flag = True

        return response, err_flag
    

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def _build_prompt_classify(self, demo, desc, graph, specialty_str):
        return f"""
请根据患者的基本信息和以下症状之间的关系，推测最可能的就诊科室。

患者信息：
{demo}

症状图谱关系如下：
{graph}

请从**{specialty_str}**中选出最合适的科室名称, **格式以外的信息请勿输出**。
输出格式如下：
Specialty: <中文科室>
"""
    
    def _build_prompt_symptom_graph(self, symptom_list, demographics):
        # Build the prompt for relationship inference
        return f"""
你是一个医学助手，请基于以下症状，推测它们之间的可能关系（例如：可能共病、因果关系、同属系统、无明显关联等），并输出有向边的三元组。

病人信息：
性别: {demographics.get("sex", "未知")}
年龄: {demographics.get("age", "未知")}
职业: {demographics.get("occupation", "未知")}

输出格式为JSON数组，每个元素是一个三元组: [源症状, 目标症状, 关系]，如:
[
["左侧肢体无力", "口干", "可能共病"],
["左侧肢体无力", "便秘", "神经系统相关可能性"]
]

你需要处理的症状列表如下, 并且输出必须是中文:
{symptom_list}
"""

    def _build_prompt_extract_structured_info(self, description):
        return f"""
你是一个医学助手。请从以下文本中提取结构化信息，返回一个JSON格式，包含两个部分："demographics" 和 "symptoms"。
每个symptom应当包含: "term", 可选的"onset", "duration", "location", "modifier", "response_to_rest"，若信息缺失可省略。

**除了JSON内容以外, 其他东西不要包括在输出中**
输出格式如下:
{{
"demographics": {{
    "sex": "...",
    "age": ...,
    "occupation": "..."
}},
"symptoms": [
    {{
    "term": "...",
    "onset": "...",
    "location": "...",
    "modifier": "...",
    "response_to_rest": "..."
    }},
    ...
]
}}

你需要处理的输入如下, 并且输出必须是中文:
{description}
"""



