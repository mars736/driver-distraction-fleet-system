from PIL import Image

import os
import re

import torch

from train import CLASS_NAMES,  Resnet50, build_transforms

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 一次性列出并排序，避免每次 __getitem__ 都重新 os.listdir（原来很慢）
        def num_key(fname):
            m = re.search(r"\d+", fname)
            return int(m.group()) if m else 0
        self.files = sorted(os.listdir(self.root), key=num_key)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return fname, img

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Resnet50(10).to(device)
    ckpt = torch.load(r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\output\best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _,val_tf=build_transforms()
    dataset=TestDataset(r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\state-farm-distracted-driver-detection\imgs\test", val_tf)
    model.eval()
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            fname, img = dataset[i]
            img = img.unsqueeze(0).to(device)
            logits, _ = model(img)
            pred = logits.argmax(dim=1).item()
            line = f"{fname} -> {pred} -> {CLASS_NAMES[f'c{pred}']}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
