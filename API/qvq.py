from openai import OpenAI
import tqdm
import json
from pathlib import Path
from PIL import Image
import base64
import os
import io
import time
import datetime

# 读取 API Key 和 Base URL
with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()
API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 将图像编码为 base64（自动压缩超过10MB的图像）
def encode_image(image_path, max_bytes=10 * 1024 * 1024):
    with open(image_path, "rb") as f:
        image_data = f.read()
    if len(image_data) <= max_bytes:
        return base64.b64encode(image_data).decode("utf-8")

    # 超过10MB，进行压缩
    image = Image.open(image_path).convert("RGB")
    quality = 95
    while quality >= 10:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        compressed_data = buffer.getvalue()
        if len(compressed_data) <= max_bytes:
            return base64.b64encode(compressed_data).decode("utf-8")
        quality -= 5
    raise ValueError(f"Cannot compress image {image_path} under 10MB limit.")

# 调用 API，支持流式返回推理内容
def call_api(question, image_path, system_prompt):
    base64_image = encode_image(image_path)
    reasoning_content = ""
    answer_content = ""
    is_answering = False

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}" }},
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
You are a multilingual visual question answering expert.

Your task is to analyze the image and answer the user's question **based on the image content only**.

Only output the **key answer** to the question, and **enclose it in double quotes**, like: "your answer".

Do not include explanations, full sentences, or redundant information. Only the most concise answer.

Example:
Question: What is the price of the item?
Answer: "€3.50"

Now, answer the following question:
"""

# 执行推理
def run_vqa_inference(json_file):
    data = json.load(open(json_file, 'r'))
    output = []
    sleep_times = [5, 10, 20]

    for idx, item in enumerate(tqdm.tqdm(data)):
        image_path = os.path.join(image_folder, item["image"])
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
            "image_path": item["image"],
            "question": question,
            "answer": item.get("answer", ""),
            "reasoning": reasoning,
            "model_output": answer
        })

    output_path = os.path.join(output_dir, f"{model_name}_{Path(json_file).stem}_results.json")
    json.dump(output, open(output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"✅ Saved output to: {output_path}")

# 主程序入口
if __name__ == "__main__":
    today = datetime.date.today()

    model_name = "qvq-max"
    image_folder = "/mnt/workspace/xintong/jlq/dataset/MTVQA-Test/"
    output_dir = f"/mnt/workspace/xintong/jlq/All_result/MTVQA/qvq-mtvqa-test-results-{today}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_file = "../data/mtvqa_all_data_test_filtered_cleaned_tomodels.jsonl"

    run_vqa_inference(input_file)
