# -*- coding: utf-8 -*-
"""
从已经训练好的 best.pth 重建 FAISS 索引和 label_map.json，不需要重新训练。

用法（在 driver-distraction-fleet-system 目录下）：
    python build_faiss.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import faiss
from torch.utils.data import DataLoader

from train import CLASS_NAMES, DriverDataset, Resnet50, build_transforms


def main():
    base = r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50"
    ckpt_path = os.path.join(base, "output", "best.pth")
    csv_path = os.path.join(base, "state-farm-distracted-driver-detection", "driver_imgs_list.csv")
    imgs_root = os.path.join(base, "state-farm-distracted-driver-detection", "imgs", "train")
    out_dir = os.path.join(base, "output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1. 建模型并加载最佳权重
    model = Resnet50(10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载 best.pth，val_acc={ckpt.get('val_acc')}")

    # 2. 全部训练图 + 固定预处理（与训练脚本一致）
    _, val_tf = build_transforms()
    df = pd.read_csv(csv_path)
    all_ds = DriverDataset(df, imgs_root, val_tf)
    all_loader = DataLoader(all_ds, batch_size=32, shuffle=False, num_workers=0)

    # 3. 抽 2048 维特征
    feats = []
    with torch.no_grad():
        for images, _ in all_loader:
            images = images.to(device)
            _, feat = model(images)
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats).astype(np.float32)
    print("特征形状:", feats.shape)

    # 4. 建索引，用 Python open 写，避开 faiss 的中文路径问题
    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(feats)
    faiss_path = os.path.join(out_dir, "faiss.index")
    with open(faiss_path, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())
    print(f"[FAISS] 已保存: {faiss_path}")

    # 5. 保存类别映射
    label_map = {f"c{i}": CLASS_NAMES[f"c{i}"] for i in range(10)}
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print("label_map.json 已保存")


if __name__ == "__main__":
    main()
