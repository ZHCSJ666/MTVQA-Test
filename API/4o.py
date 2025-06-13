import openai
import tqdm
import json
from pathlib import Path
from PIL import Image
import os
import base64
import time
import datetime

# 读取 API Key 和 Base URL
with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

openai.api_key = API_KEY
openai.base_url = BASE_URL

# 图像编码为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# API 调用函数
def call_api(question, image_path, system_prompt):
    base64_image = encode_image(image_path)
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": question}
                ],
            }
        ],
    )
    return response.choices[0].message.content

# 系统提示词
system_prompt = """
You are a multilingual visual reasoning expert.
Answer the question based on the image and the text.
Highlight the key reasoning result using double quotes, like: "the answer".
"""

# 推理主函数
def run_vqa_inference(json_file):
    data = json.load(open(json_file, 'r'))
    output = []
    sleep_times = [5, 10, 20]

    for idx, item in enumerate(tqdm.tqdm(data)):
        image_path = os.path.join(image_folder, item["image_path"])
        question = item["question"]

        for sleep_time in sleep_times:
            try:
                result = call_api(question, image_path, system_prompt)
                break
            except Exception as e:
                print(f"Error: {e}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        else:
            result = "ERROR"

        output.append({
            "id": item.get("id", idx),
            "lang": item.get("lang", ""),
            "image_path": item["image_path"],
            "question": question,
            "answer": item.get("answer", ""),
            "model_output": result
        })

    output_path = os.path.join(output_dir, f"{model_name}_{Path(json_file).stem}_results.json")
    json.dump(output, open(output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved output to {output_path}")

# 主程序入口，使用固定输入路径
if __name__ == "__main__":
    today = datetime.date.today()

    model_name = "gpt-4o-2024-11-20"
    image_folder = "/mnt/workspace/xintong/jlq/dataset/MTVQA-Test/"
    output_dir = f"/mnt/workspace/xintong/jlq/All_result/MTVQA/gpt4o-mtvqa-test-results-{today}/"
    Path(output_dir).mkdir(exist_ok=True)

    # ✅ 写死输入文件路径
    input_file = "../data/mtvqa_all_data_test_filtered_cleaned_tomodels.jsonl"

    run_vqa_inference(input_file)
