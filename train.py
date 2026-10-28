import os
import pandas as pd
import torch
import torchvision
import faiss
import numpy as np
from PIL import Image     #python PIL的经典一个图像类，torchvision里对图片预处理都是对这个类进行预处理
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms     #图形预处理模块

CLASS_NAMES={
    "c0":"正常开车",
    "c1":"右手玩手机",
    "c2":"右手打电话",
    "c3":"左手玩手机",
    "c4":"左手打电话",
    "c5":"操作中控屏",
    "c6":"喝水",
    "c7":"向后伸手",
    "c8":"整理头发/化妆",
    "c9":"与乘客交谈"
}

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
    :param loader: 加载器，加载器给出的images一次就是一批，当然是Tensor格式的
    :param device: nn.device，里面装了是cpu还是gpu
    :param criterion: 损失函数，后面会往这里传nn.CrossEntropyLoss交叉熵
    :return:
    """
    model.eval()#切换为测试，放弃dropout
    total_loss=0.0#整个测试集的损失和
    all_pred=[]#整个测试集的所有预测标签
    all_true=[]#整个测试集的所有真实标签
    for image,label in loader:
        images,labels = image.to(device),label.to(device)#把images和labels都放进gpu，模型在哪，数据集就必须在哪
        logits,_=model(images) #_只是来占位的，反正feature我现在也不要
        loss=criterion(logits,labels)#计算损失，此处的输出仅为一个数，交叉熵本身的公式对单个样本的计算就是单个数字，但是这里是一个batch，也是一个数的原因是会对这个批次的所有样本取平均
        total_loss+=loss.item()*images.size(0)#平均值*样本数=该批次的总loss，images.size(0)就是计算images第0维的样本数，即该batch有几个样本，不乘一个常量的原因是loader在取图片取到后面时，如果数据集的图片不够，该batch的样本数就会变
        all_pred.append(logits.argmax(1).cpu().numpy())#前向传播得到的logits是一个二维的Tensor，argmax(1)其中1的含义是从第2维角度（即每一行）找到最大值的下标，封装为一个列表
        all_true.append(labels.cpu().numpy())#把labels从gpu转到cpu，再转为numpy数组，封装为一个列表，塞进这个大列表
    #至此，前向传播完成，拿到了最基本的三个值:个测试集的total_loss、所有样本的预测标签all_pred（注意这里一个批次的样本的预测标签是放在一个小列表里面的，这个小列表又是外面这个大列表的一个元素），all_true存的是真实标签（里面的结构跟all_pred类似）
    pred=np.concatenate(all_pred)#效果就是把里面的小列表拆开
    true=np.concatenate(all_true)
    acc=float((pred==true).mean())#(pred==true)这里返回的是bool ndarray，取平均值就是准确率了
    avg_loss=total_loss/len(loader.dataset)#dataset是loader里的存的一个对象

    #至此，我们就得到了测试集的损失和准确率，再来一个混淆矩阵
    cm=np.zeros((10,10),dtype=int)
    for t,p in zip(true,pred):
        cm[t,p]+=1#只有在对角线上才是预测正确
    return acc,avg_loss,cm




def print_report(cm):
    """
    打印每个类别的精确率，召回率，F1值
    TP（True Positive，真正例）：真实是 c1、也预测成 c1 的数量 = cm[1,1] = 8。
    FP（False Positive，假正例）：真实不是 c1，却被预测成 c1 的数量 = 第 1 列的总和减去 TP = (1+8+0) - 8 = 1。也就是「冤枉」了多少别的类。
    FN（False Negative，假负例）：真实是 c1，却被预测成别的类 = 第 1 行的总和减去 TP = (2+8+1) - 8 = 3。也就是「漏掉」了多少 c1。
    精确率 Precision = TP / (TP + FP)：在「所有被预测为 i 的样本」里，真正是 i 的比例。查得准不准。
    召回率 Recall = TP / (TP + FN)：在「所有真实的 i 样本」里，被找出来的比例。查得全不全。
    F1 = 2 × P × R / (P + R)：精确率和召回率的调和平均，两者都高 F1 才高。
    :param cm:
    :return:
    """
    print("类别  精确率  召回率  F1")
    for i in range(cm.shape[0]):#遍历每个类别
        tp=cm[i,i]
        fp=cm[i,i]-tp
        fn=cm[i,i]-fp
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1=2*precision*recall/(precision+recall)
        print(f"c{i:<2} {precision:6.3f} {recall:6.3f} {f1:6.3f}  ({CLASS_NAMES[f'c{i}']})")





def build_faiss_index(model,loader,device,out_path):
    """
    构建faiss索引
    :param model:
    :param loader:
    :param device: 设备对象
    :param out_path: 输出路径
       :return:
    """
    model.eval()#切换为测试，放弃dropout,放弃更新BN层的缓存，防止污染数据
    feats=[]#保存所有features
    with torch.no_grad():#with后面跟的是一个实现了__enter__和__exit__方法的对象，这里实现的是torch.no_grad()，它的作用是在with语句块中，不计算梯度，不更新参数，不记录历史记录
        for image,_ in loader:
            images=image.to(device)
            _,feat=model(images)
            feats.append(feat.cpu().numpy())
    feats=np.concatenate(feats).astype(np.float32)#展开所有features（当然是二维），转换为float32类型，因为faiss索引要求是float32类型，不转大概率是float64
    #关于concatenate（）这个函数：
    # 拼接前：
#   batch1: (32,  2048)
#   batch2: (32,  2048)
#   batch3: (12,  2048)   ← 最后一批可能不满
#
# 拼接后：
#   feats:  (76,  2048)   ← axis=0 首尾相接，列数不变
    index=faiss.IndexFlatL2(feats.shape[1])#创建一个L2距离索引，告诉它我们的特征维度是多少（列数）
    index.add(feats)#把所有特征加进来
    faiss.write_index(index,out_path)#保存进磁盘
    print(f"[FAISS] 已保存索引，样本数={feats.shape[0]}，维度={feats.shape[1]} -> {out_path}")





def main():
    pass









