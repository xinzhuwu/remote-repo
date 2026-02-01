import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet101_Weights
"""载入CNN图像分类器,图像处理的标准化normalize的作用是将像素值从[0,255]或[0,1]范围转换为接近标准正态分布的数值，
加速模型收敛，使输入分布与预训练模型训练时的数据分布一致"""
model=torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
print(model)
"""整体表示提取器为删除最后一个fc层的Resnet101"""
global_representation_extractor=torch.nn.Sequential(*(list(model.children())[:-1]))
"""网格表示提取器为删除最后两层的Resnet101"""
grid_representation_extractor=torch.nn.Sequential(*(list(model.children())[:-2]))
"""图像预处理流程"""
preprocess_image=transforms.Compose([
    transforms.Resize(256,),   #将短边缩放到256像素，等比例缩放
    transforms.CenterCrop(224),  #从图像中心裁剪出224 * 224区域
    transforms.ToTensor(),#将图像或数组转换为张量，数据类型转换为float32， 像素值从[0,255]归一化到[0.0, 1.0],维度顺序从(H,W,C)转到(C,H,W) 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
"""读取图像"""
image=Image.open("C:/Users/wuxinzhu/Desktop/P02650_EMS_mix_all_rug_CLT_3203.jpg")
"""执行图像预处理"""
img=preprocess_image(image).unsqueeze(0) #将单张图片转换成batch格式
"""提取整体表示"""
global_representation=global_representation_extractor(img)
"""提取网格表示"""
grid_representation=grid_representation_extractor(img)
print("整体表示的形状：", global_representation.shape)
print("网格表示的形状: ", grid_representation.shape)

"""
AdaptiveAvgPool2d的作用是将任意大小的二维特征图压缩成固定大小(1 * 1)
"""


