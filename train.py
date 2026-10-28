import os
import pandas as pd
import torch
import torchvision
from PIL import Image     #python PIL的经典一个图像类，torchvision里对图片预处理都是对这个类进行预处理
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms     #图形预处理模块

#ResNet50_Weights.IMAGENET1K_V2 的标准化参数:平均值和标准差
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

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
        image= Image.open(path).convert("RGB")#以RBG格式打开图片，这里返回的不是张量，如果不走transform就一定在loader那里会出问题，因为loader只吃张量
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
        feature=x.flatten(1)#保留batch_size，展平其他维度（保留第0维）
        logits=self.fc(feature)
        return logits,feature



def build_transforms():
    """
    返回的是实现了__call__魔法方法的可调用对象，传给Dataset里的transform参数
    :return:
    """
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),#把图片像素变成224*224
        transforms.RandomAffine(degrees=10,translate=(0.05,0.05),scale=(0.9,1.1),shear=10),#degrees旋转角度，translate横向和纵向都在 [-5%, +5%] 的图像尺寸范围内随机平移,scale缩放,shear剪切把图片推倒
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    return train_tf,test_tf



@torch.no_grad()
def evaluate(model,loader,device,criterion):
    """

    :param model: resnet模型
    :param loader: 加载器
    :param device: nn.device，里面装了是cpu还是gpu
    :param criterion: 损失函数，后面会往这里传nn.CrossEntropyLoss交叉熵
    :return:
    """
    model.eval()#切换为测试，放弃dropout
