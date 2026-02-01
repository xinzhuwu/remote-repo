from argparse import Namespace 
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet152_Weights, VGG19_Weights
from PIL import Image
import os
import random
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
from datasets import load_dataset

"""
1、下载数据集
2、整理数据集
"""
def create_dataset(dataset="flickr8k", captions_per_image=5, min_word_count=5, max_len=30):
    """
    数据集整理
    dataset:数据集名称
    captions_per_image :每张图像对应的文本描述数
    min_word_count:仅考虑在数据集中至少出现5次的词
    max_len:文本描述包含的最大单词数，如果文本描述超过该值，则截断
    输出：一个词典文件，三个数据集文件
    """
    kerpathy_json_path="./data/dataset_flickr8k.json"
    image_folder="data/Images"
    output_folder="data/datasets"
    with open (kerpathy_json_path) as j:
        data=json.load(j)
    image_paths=defaultdict(list)
    image_captions=defaultdict(list)
    vocab=Counter()
    for img in data["images"]:
        split=img["split"]
        captions=[]
        for c in img["sentences"]:
            "更新词频，测试集在训练过程中未见数据集，不能统计"
            if split != "test":
                vocab.update(c["tokens"])
            #不统计超过最大长度的词
            if len (c["tokens"]) <= max_len:
                captions.append(c["tokens"])
        if len(captions)==0:
            continue
        path=os.path.join(image_folder, img["filename"])
        image_paths[split].append(path)
        image_captions[split].append(captions)
    #创建词典，增加占位标识符<pad>, 未登录词标识符<unk>, 句子首尾标识符<start> <end>
    words=[w for w in vocab.keys() if vocab[w] >= min_word_count]
    vocab={k:v+1   for v, k  in enumerate(words)}
    vocab["<pad>"]=0
    vocab["<unk>"]=len(vocab)
    vocab["<start>"]=len(vocab)
    vocab["<end>"]=len(vocab)
    #存储词典
    with open(os.path.join(output_folder, "vocab.json"), "w") as fw:
        json.dump(vocab, fw, indent=2, ensure_ascii=False)
    for split in image_paths:
        imgpaths=image_paths[split]
        imcaps=image_captions[split]
        """图片和文本的描述是按照顺序对应的"""
        enc_captions=[]
        for i, path in enumerate(imgpaths):
            #合法性检查，检查图像是否可以被解析
            img=Image.open(path)
            #如果该图像对应的描述不足，则补足
            if len(imcaps[i]) < captions_per_image:
                filled_num=captions_per_image - len(imcaps[i])
                captions=imcaps[i] + [random.choice(imcaps[i]) for _ in range(filled_num)]
            #random.choice是按照均匀分布随机采样，random.sample是随机选择k个不重复的
            #如果该图像对应的描述超了，则随机采样
            else:
                captions=random.sample(imcaps[i], k=captions_per_image)
            assert len(captions) ==  captions_per_image
            #按照图片顺序依次处理5个文本描述
            for j, c in enumerate(captions):
                #对文本描述进行编码
                enc_c=[vocab["<start>"]] + [vocab.get(word, vocab["<unk>"]) for word in c] + [vocab["<end>"]]
                enc_captions.append(enc_c)

        #合法性检查
        assert len(imgpaths) * captions_per_image == len(enc_captions)
        #存储数据
        data={
            "IMAGES": imgpaths,
            "CAPTIONS": enc_captions
        }
        with open(os.path.join(output_folder, split+"_data.json"), "w") as fw:
            json.dump(data, fw)
if __name__=="__main__":
    create_dataset()
