from huggingface_hub import hf_hub_download
import zipfile
import os
from tqdm import tqdm

# 下载 zip 文件（自动缓存）
print("📥 正在从 HuggingFace 下载数据集...")
zip_path = hf_hub_download(
    repo_id="ZHSCJ/MTVQA-Test-Errors",
    filename="mtvqa_all_samples_test.zip",
    repo_type="dataset"
)

# 解压目标目录
target_dir = "/mnt/workspace/xintong/jlq/dataset/MTVQA-Test/"
os.makedirs(target_dir, exist_ok=True)

# 解压 zip 文件并显示进度条
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    members = zip_ref.infolist()
    print(f"📦 开始解压 {len(members)} 个文件到 {target_dir} ...")
    for member in tqdm(members, desc="解压进度"):
        zip_ref.extract(member, target_dir)

print(f"✅ 数据已成功下载并解压到：{target_dir}")
