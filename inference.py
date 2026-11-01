# -*- coding: utf-8 -*-
"""
FastAPI 推理子服务 —— 对应后端架构的【六、识别调度与推理】和【七、OOD 拒识】两个模块。

职责：
    1. 启动时加载 best.pth（ResNet50 权重）和 faiss.index（FAISS 索引）。
    2. 接收 SpringBoot 用 multipart 传来的 JPEG 帧（一批）。
    3. 做 ResNet50 推理（logits + 2048 特征）。
    4. 做 OOD 三段式：置信度 -> Energy -> FAISS KNN。
    5. 返回每个帧的 类别 / 置信度 / Energy / KNN距离 / 是否OOD，不返回 2048 特征。

运行（在 driver-distraction-fleet-system 目录下）：
    pip install fastapi uvicorn python-multipart
    uvicorn inference:app --host 0.0.0.0 --port 8000

测试（另开一个终端）：
    curl -F "files=@C:/某路径/a.jpg" -F "files=@C:/某路径/b.jpg" http://127.0.0.1:8000/predict
    或者：python test_client.py
"""

import io
from contextlib import asynccontextmanager

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel#pydantic 是一个 Python 的数据验证和设置管理库，可数据解析、序列化等

import faiss

# 复用训练脚本里的模型结构、预处理、类别名
from train import CLASS_NAMES, Resnet50, build_transforms

# ------- 路径配置（按需改） -------
BASE = r"C:\Users\jianxuyang\Desktop\codex工作夹2\resnet50"
MODEL_PATH = BASE + r"\output\best.pth"
FAISS_PATH = BASE + r"\output\faiss.index"

# ------- OOD 阈值（TODO：这些是占位值，必须标定后才能用，见文末说明） -------
CONF_THRESHOLD = 0.6327        # 置信度 max_p 低于此值 -> OOD
ENERGY_THRESHOLD = -2.8681    # Energy 高于此值 -> OOD
KNN_THRESHOLD = 280.5992       # KNN 平均 L2 平方距离高于此值 -> OOD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
index = None
transform = None


def load_faiss(path):
    """读取 faiss.index（build_faiss.py 里是用 serialize_index 存的字节）。"""
    with open(path, "rb") as f:
        return faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型和索引。"""
    global model, index, transform

    # 注意：Resnet50.__init__ 会先加载 ImageNet 预训练权重（已在本地缓存），再被 best.pth 覆盖
    model = Resnet50(10).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    index = load_faiss(FAISS_PATH)
    _, transform = build_transforms()  # 取固定的 val/test transform（Resize224 + ToTensor + Normalize）

    print(f"[启动完成] device={device}，FAISS 样本数={index.ntotal}，维度={index.d}")
    yield


app = FastAPI(title="驾驶员违规行为识别推理子服务", lifespan=lifespan)


class FrameResult(BaseModel):#提前定义成员变量，等会返回给springboot
    pred: int
    class_name: str
    max_p: float
    energy: float
    knn_dist: float
    is_ood: bool


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")#FastAPI 在启动时调 inspect.signature(predict)，从返回的签名里取 signature.parameters['files'].default，也就是代码里写的 File(...) 标记对象。读到了什么：param.default = File(...) 标记 + param.annotation = list[UploadFile]。
#之后：FastAPI 根据这两样建好解析器；每次请求只是用这个解析器去读 multipart，然后把结果传给 predict(files=[...])。
def predict(files: list[UploadFile] = File(...)):#...类型是 ellipsis,传进去了之后fastapi会知道default参数是ellipsis，那就会强制要求files必须传文件给后端
    """接收一批 JPEG 帧，批量推理 + OOD。返回结果数组，顺序与上传顺序一致。
    | 参数来源 | FastAPI | Spring Boot |
|---|---|---|
| URL 路径 `/user/{id}` | `Path(...)` | `@PathVariable` |
| 查询串 `?a=1` | `Query(...)` | `@RequestParam` |
| JSON 请求体 | `Body(...)` / Pydantic 模型 | `@RequestBody` |
| 表单字段 | `Form(...)` | `@RequestParam` / `@ModelAttribute` |
| 上传文件 | `File(...)` + `UploadFile` | `@RequestPart` + `MultipartFile` |
| 请求头 | `Header(...)` | `@RequestHeader` |
| Cookie | `Cookie(...)` | `@CookieValue` |
| 依赖注入 | `Depends(...)` | `@Autowired` / `@Resource` |
    """
    batch = []
    for f in files:
        data = f.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")#io.BytesIO 是 Python 标准库 io 模块中的一个类，它创建一个内存中的二进制流（字节流）对象，可以像操作文件一样操作内存中的字节数据。本来Image.open() 期望接收一个文件路径或文件对象，但是这样子的话传入的字节流文件就必须先保存在磁盘再让open()去读，io效率太慢，io.BytesIO(data) 可以将这些字节包装成一个内存中的"虚拟文件"，直接传给open()而不走磁盘
        batch.append(transform(img))
    if not batch:
        return {"results": []}#一张图都没传直接返回空结果

    x = torch.stack(batch).to(device)  # (N, 3, 224, 224) torch.stack 是把一堆形状相同的张量叠起来，多出一个新维度。

    # 2. 推理（模块六）：logits + 2048 特征
    with torch.no_grad():
        logits, feat = model(x)

    probs = logits.softmax(dim=1)
    max_p, pred = probs.max(dim=1)            # 置信度 + 预测类别
    energy = -torch.logsumexp(logits, dim=1)  # Energy = -logsumexp(logits)，energy越负，logits内的个元素之间差值越大，说明模型确信这张图片就是哪一个类别，就不会拒识

    # 3. OOD KNN（模块七）：K=5，平均 L2 平方距离
    D, _ = index.search(feat.cpu().numpy().astype("float32"), 5)
    knn_dist = D.mean(axis=1)
    """
    feat.cpu().numpy()：特征从 GPU 搬到 CPU、转成 numpy。
    .astype("float32")：转成 float32（FAISS 要求）
    index.search(特征, 5)：在 FAISS 索引里找最近 5 个邻居，返回 (距离, 邻居编号)。D 是距离（shape N×5），_ 丢掉编号。
    D.mean(axis=1)：对每一行（每张图）的 5 个距离求平均，得到平均 L2 平方距离。
    D的形状：
             邻居1  邻居2  邻居3  邻居4  邻居5
帧1      0.1    0.2    0.3    0.4    0.5
帧2      1.0    1.1    1.2    1.3    1.4"""

    # 4. 三段式判定
    results = []
    for i in range(x.size(0)):
        is_ood = (
            float(max_p[i]) < CONF_THRESHOLD
            or float(energy[i]) > ENERGY_THRESHOLD
            or float(knn_dist[i]) > KNN_THRESHOLD
        )
        results.append(FrameResult(
            pred=int(pred[i]),
            class_name=CLASS_NAMES[f"c{int(pred[i])}"],
            max_p=round(float(max_p[i]), 4),
            energy=round(float(energy[i]), 4),
            knn_dist=round(float(knn_dist[i]), 4),
            is_ood=is_ood,
        ))
    return {"results": results}


"""
阈值标定方法：
    把训练集（22424 张）全过一遍模型，收集每张的 max_p、energy、knn_dist 的分布，
    然后取一个百分位，例如：
        CONF_THRESHOLD   = max_p 的第 5 百分位（95% 的训练样本 max_p 都高于它）
        ENERGY_THRESHOLD = energy 的第 95 百分位
        KNN_THRESHOLD    = knn_dist 的第 95 百分位
    这样"见过的样本"大概率能通过，"没见过的 OOD 样本"会超过阈值被拒识。
"""
