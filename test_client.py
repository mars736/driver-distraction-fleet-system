# -*- coding: utf-8 -*-
"""本地测试 FastAPI 推理服务的脚本：传几张图，打印返回结果。

用法：
    1. 先启动服务：uvicorn inference:app --host 0.0.0.0 --port 8000
    2. 另开一个终端：pip install requests
    3. python test_client.py
"""

import requests

URL = "http://127.0.0.1:8000/predict"

# 挑几张测试集图片（路径/数量按需改，这几张都存在）
imgs = [
    r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\state-farm-distracted-driver-detection\imgs\test\img_0.jpg",
    r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\state-farm-distracted-driver-detection\imgs\test\img_1.jpg",
    r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\state-farm-distracted-driver-detection\imgs\test\img_2.jpg",
    r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50\state-farm-distracted-driver-detection\imgs\test\img_3.jpg",
]

opened = [open(p, "rb") for p in imgs]
try:
    files = [("files", (p.split("\\")[-1], f, "image/jpeg")) for p, f in zip(imgs, opened)]
    r = requests.post(URL, files=files)
    print("HTTP", r.status_code)
    print(r.json())
finally:
    for f in opened:
        f.close()
