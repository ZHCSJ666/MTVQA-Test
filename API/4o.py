from openai import OpenAI
import tqdm
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import base64
import time
import datetime

# 读取 API Key 和 Base URL
with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()
API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

# 初始化 OpenAI 客户端（兼容 DashScope）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 图像编码为 Base64（自动压缩到 10MB 以下）
def encode_image(image_path):
    max_bytes = 10 * 1024 * 1024  # 10MB
    with Image.open(image_path) as img:
        img_format = "JPEG"
        quality = 95
        while quality > 10:
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format=img_format, quality=quality)
            byte_data = buffer.getvalue()
            if len(byte_data) <= max_bytes:
                return base64.b64encode(byte_data).decode("utf-8")
            quality -= 5
        raise ValueError(f"Image {image_path} cannot be compressed below 10MB")

# API 调用函数
def call_api(question, image_path, system_prompt):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
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
You are a multilingual visual question answering expert.

Analyze the image and answer the question precisely.

The answer must be inside double quotes, like: "your answer".
Only the key answer should appear inside the quotes — no extra text.
"""

# 推理主函数
def run_vqa_inference(json_file):
    data = json.load(open(json_file, 'r'))
    output = []
    sleep_times = [5, 10, 20]

    for idx, item in enumerate(tqdm.tqdm(data)):
        image_path = os.path.join(image_folder, item["image"])
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
            "image_path": item["image"],
            "question": question,
            "answer": item.get("answer", ""),
            "model_output": result
        })

    output_path = os.path.join(output_dir, f"{model_name}_{Path(json_file).stem}_results.json")
    json.dump(output, open(output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"✅ Saved output to: {output_path}")

# 主程序入口
if __name__ == "__main__":
    today = datetime.date.today()

    model_name = "gpt-4o-2024-11-20"  # ✅ 使用支持图像输入的模型
    image_folder = "/mnt/workspace/xintong/jlq/dataset/MTVQA-Test/"
    output_dir = f"/mnt/workspace/xintong/jlq/All_result/MTVQA/gpt4o-mtvqa-test-results-{today}/"
    Path(output_dir).mkdir(exist_ok=True)

    input_file = "../data/mtvqa_all_data_test_filtered_cleaned_tomodels.jsonl"

    run_vqa_inference(input_file)
