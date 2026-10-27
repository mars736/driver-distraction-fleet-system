import os
import pandas as pd
import torch
import torchvision
from PIL import Image     #python PIL的经典一个图像类，torchvision里对图片预处理都是对这个类进行预处理
from torch import nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms     #图形预处理模块



class DriverDataset(Dataset):
    def __init__(self, df,imgs_root,transform=None):
        self.df = df
        self.imgs_root = imgs_root
        self.transform = transform
        self.class_to_idx = {f"c{i}": i for i in range(10)}  #c0->0,c1->1,c2->2...

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: image返回PIL图片,label返回该图片的标签
        """
        row=self.df.iloc[idx]    #iloc函数提取出DataFrame的一行
        path = os.path.join(self.imgs_root,row["filename"])#把路径拼出来，准备打开该文件
        image= Image.open(path).convert("RGB")#以RBG格式打开图片
        label=self.class_to_idx[row["classname"]]
        if self.transform:
            image = self.transform(image)#该实例的transform为传入的函数，直接传函数
        return image,label




class Resnet50(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        m=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)#先把整个IMAGENET预训练的resnet50参数拿到手
        self.stem = nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool)  #把m的前几层预处理层放进Sequential容器
        self.layer1=m.layer1#四个大残差块
        self.layer2=m.layer2
        self.layer3=m.layer3
        self.layer4=m.layer4
        self.avgpool=m.avgpool
        self.fc = nn.Linear(m.fc.in_features, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        #先写到这l