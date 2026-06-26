# -*- coding: utf-8 -*-
# 一键下载 NCRDLLM 所需的 LLaMA-3.2-3B-Instruct 权重，并放到代码能找到的正确位置。
#
# 用法（在任意目录都可以，脚本会自动定位）:
#   python download_model.py
#
# 完成后，权重会落在本脚本同级目录下的:
#   ./llama3.1/LLM-Research/Llama-3___2-3B-Instruct
# 这正是 model.py / config.py 查找的本地路径，下载完直接训练即可。

import os
import sys

# 模型代码运行时用的是相对路径 ./llama3.1/...，相对的是“运行 train.py 时的目录”。
# 这里把下载目标固定到本脚本所在目录（即 train.py 同级），无论从哪里执行都正确。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "llama3.1")

# 论文默认使用 3B（隐藏维度 3072）。一般不需要改动。
MODEL_ID = "LLM-Research/Llama-3.2-3B-Instruct"

# model.py 会按顺序查找的本地路径（相对 train.py 目录）
EXPECTED_PATHS = [
    os.path.join(BASE_DIR, "llama3.1", "LLM-Research", "Llama-3___2-3B-Instruct"),
    os.path.join(BASE_DIR, "llama3.1", "LLM-Research", "Llama-3.2-3B-Instruct"),
]


def get_folder_size_gb(folder):
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def already_downloaded():
    for p in EXPECTED_PATHS:
        # 认为存在 config.json 才算下载完整
        if os.path.exists(os.path.join(p, "config.json")):
            return p
    return None


def main():
    print("=" * 56)
    print("NCRDLLM 模型下载工具  (LLaMA-3.2-3B-Instruct)")
    print("=" * 56)

    existing = already_downloaded()
    if existing:
        print("✅ 已检测到本地模型: %s" % existing)
        print("   大小: %.2f GB" % get_folder_size_gb(existing))
        print("   无需重复下载，可直接训练。")
        return 0

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("❌ 缺少 modelscope，请先安装:")
        print("   pip install modelscope")
        return 1

    print("📥 开始从 ModelScope 下载（模型约数 GB，请耐心等待，可断点续传）...")
    print("   目标目录: %s" % CACHE_DIR)
    try:
        model_path = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
    except Exception as e:
        print("❌ 下载失败: %s" % e)
        print("   请检查网络后重新运行本脚本（支持续传）。")
        return 1

    print("\n✅ 下载完成: %s" % model_path)
    print("   大小: %.2f GB" % get_folder_size_gb(model_path))

    # 校验代码能否按预期路径找到
    found = already_downloaded()
    if found is None:
        # 极少数情况下 ModelScope 目录命名不同，建立一个别名目录指向实际位置
        canonical = EXPECTED_PATHS[0]
        try:
            os.makedirs(os.path.dirname(canonical), exist_ok=True)
            if not os.path.exists(canonical):
                os.symlink(model_path, canonical)
            found = canonical
            print("🔗 已建立路径别名: %s -> %s" % (canonical, model_path))
        except Exception:
            print("⚠️  下载成功，但目录名与代码预期不同。")
            print("   实际路径: %s" % model_path)
            print("   请手动改名/移动到: %s" % canonical)
            return 1

    print("\n🎉 一切就绪！现在在本目录下运行训练:")
    print("")
    print("  python train.py --dataset miRNA-drug \\")
    print("    --use_rna_seq --use_rna_struct --use_rna_disease \\")
    print("    --use_drug_seq --use_drug_struct --use_drug_disease \\")
    print("    --use_lora --lora_r 64 --lora_alpha 64")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
