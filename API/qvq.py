import openai
import tqdm
import json
from pathlib import Path
import base64
import os
import time
import datetime

# 读取 API Key 和 Base URL
with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()
API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

openai.api_key = API_KEY
openai.base_url = BASE_URL

# 编码图像为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 调用 API，支持流式返回推理内容
def call_api(question, image_path, system_prompt):
    base64_image = encode_image(image_path)
    reasoning_content = ""
    answer_content = ""
    is_answering = False

    completion = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": question}
                ],
            },
        ],
        stream=True,
    )
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        else:
            if delta.content and not is_answering:
                is_answering = True
            answer_content += delta.content

    return reasoning_content, answer_content

# 系统提示词
system_prompt = """
You are a multilingual visual reasoning expert.
Answer the question based on the image and the text.
Highlight the key reasoning result using double quotes, like: "the answer".
"""

# 执行推理
def run_vqa_inference(json_file):
    data = json.load(open(json_file, 'r'))
    output = []
    sleep_times = [5, 10, 20]

    for idx, item in enumerate(tqdm.tqdm(data)):
        image_path = os.path.join(image_folder, item["image_path"])
        question = item["question"]

        last_error = None
        for sleep_time in sleep_times:
            try:
                reasoning, answer = call_api(question, image_path, system_prompt)
                break
            except Exception as e:
                last_error = e
                print(f"Error on {idx}: {e}. Retry after sleeping {sleep_time} sec...")
                time.sleep(sleep_time)
        else:
            print(f"Skipping {idx} after retries")
            reasoning, answer = "", ""
            item["error"] = str(last_error) if last_error else ""

        output.append({
            "id": item.get("id", idx),
            "lang": item.get("lang", ""),
            "image_path": item["image_path"],
            "question": question,
            "answer": item.get("answer", ""),
            "reasoning": reasoning,
            "model_output": answer
        })

    output_path = os.path.join(output_dir, f"{model_name}_{Path(json_file).stem}_results.json")
    json.dump(output, open(output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved output to {output_path}")

# 主程序入口（写死输入路径）
if __name__ == "__main__":
    today = datetime.date.today()

    model_name = "qvq-max"
    image_folder = "/mnt/workspace/xintong/jlq/dataset/MTVQA-Test/"
    output_dir = f"/mnt/workspace/xintong/jlq/All_result/MTVQA/qvq-mtvqa-test-results-{today}/"
    Path(output_dir).mkdir(exist_ok=True)

    # ✅ 写死输入 JSON 路径
    input_file = "../data/mtvqa_all_data_test_filtered_cleaned_tomodels.jsonl"

    run_vqa_inference(input_file)
