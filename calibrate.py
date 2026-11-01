# -*- coding: utf-8 -*-
"""
标定 OOD 阈值：在验证司机上统计 max_p / energy / knn_dist 的分布，给出建议阈值。

原理：
    1. 用 21 个训练司机建一份【临时】FAISS 索引（只存在内存里，不写盘，脚本结束自动释放）。
    2. 在 5 个验证司机（不在临时索引里）上跑模型，收集三个指标。
    3. 取百分位作为阈值：
        - 置信度阈值 = max_p 的第 5 百分位（低于它判 OOD）
        - Energy 阈值 = energy 的第 95 百分位（高于它判 OOD）
        - KNN 阈值   = knn_dist 的第 95 百分位（高于它判 OOD）
    4. 打印建议阈值，填回 inference.py。

运行（在 driver-distraction-fleet-system 目录下）：
    python calibrate.py
"""

import numpy as np
import pandas as pd
import torch
import faiss
from torch.utils.data import DataLoader

from train import DriverDataset, Resnet50, build_transforms


# ------- 路径配置 -------
BASE = r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50"
MODEL_PATH = BASE + r"\output\best.pth"
CSV_PATH = BASE + r"\state-farm-distracted-driver-detection\driver_imgs_list.csv"
IMGS_ROOT = BASE + r"\state-farm-distracted-driver-detection\imgs\train"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1. 加载模型
    model = Resnet50(10).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_tf = build_transforms()  # 固定预处理

    # 2. 按司机划分（和 train.py 一致：前 5 个司机做验证）
    df = pd.read_csv(CSV_PATH)
    subjects = sorted(df["subject"].unique())
    val_subjects = subjects[:5]
    train_subjects = subjects[5:]
    train_df = df[df["subject"].isin(train_subjects)]
    val_df = df[df["subject"].isin(val_subjects)]
    print(f"训练司机 {len(train_subjects)} 个，验证司机 {len(val_subjects)} 个：{val_subjects}")

    # 3. 用【训练司机】建临时索引（只在内存，不写盘）
    train_ds = DriverDataset(train_df, IMGS_ROOT, val_tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.to(device)
            _, f = model(img)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats).astype("float32")
    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(feats)
    print(f"临时索引已建：{feats.shape[0]} 条，维度 {feats.shape[1]}（内存中，结束自动释放）")

    # 4. 在【验证司机】上统计三个指标
    val_ds = DriverDataset(val_df, IMGS_ROOT, val_tf)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    maxp_list, energy_list, knn_list = [], [], []
    with torch.no_grad():
        for img, _ in val_loader:
            img = img.to(device)
            logits, f = model(img)

            probs = logits.softmax(dim=1)
            max_p = probs.max(dim=1).values            # 置信度
            energy = -torch.logsumexp(logits, dim=1)   # Energy

            D, _ = index.search(f.cpu().numpy().astype("float32"), 5)  # K=5
            knn = D.mean(axis=1)                       # 平均 L2 平方距离

            maxp_list.append(max_p.cpu().numpy())
            energy_list.append(energy.cpu().numpy())
            knn_list.append(knn)

    max_p = np.concatenate(maxp_list)
    energy = np.concatenate(energy_list)
    knn = np.concatenate(knn_list)

    # 5. 看分布
    print("\n=== 分布统计（验证集）===")
    for name, arr in [("max_p", max_p), ("energy", energy), ("knn_dist", knn)]:
        print(f"{name}: 均值={arr.mean():.4f}  最小={arr.min():.4f}  最大={arr.max():.4f}")

    conf_th = np.percentile(max_p, 5)     # 95% 的样本 max_p 都高于它
    energy_th = np.percentile(energy, 95)  # 95% 的样本 energy 都低于它
    knn_th = np.percentile(knn, 95)        # 95% 的样本 knn_dist 都低于它

    print("\n=== 建议阈值（复制填回 inference.py）===")
    print(f"CONF_THRESHOLD = {conf_th:.4f}")
    print(f"ENERGY_THRESHOLD = {energy_th:.4f}")
    print(f"KNN_THRESHOLD = {knn_th:.4f}")


if __name__ == "__main__":
    main()
