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



"""
3、定义数据集类
"""
class ImageTextDataset(Dataset):
    def __init__(self, dataset_path, vocab_path, split, captions_per_image=5, max_len=30, transforms=None):
        """
        参数说明:
        dataset_path: json格式数据文件路径
        vocab_path: 格式词典文件路径
        split:train, val, test
        transforms:图像预处理方法
        """
        super(ImageTextDataset, self).__init__()
        self.split=split
        assert self.split in ["train", "test", "val"]
        self.cpi=captions_per_image
        self.max_len=max_len
        #载入数据集
        with open(dataset_path, "r") as f:
            self.data=json.load(f)
        #载入词典
        with open(vocab_path, "r") as f:
            self.vocab=json.load(f)
        #Pytorch图像预处理流程
        self.transforms=transforms
        #数据量计算使用captions  数量上关系 captions = imgs * cpi,因为是使用caption作为数量总量，所以
        #是每隔cpi个取一张图片
        self.dataset_size=len(self.data["CAPTIONS"])
    def __getitem__(self,i):
        """第i个文本描述对应的是第i //5张图片"""
        img=Image.open(self.data["IMAGES"][i// self.cpi]).convert("RGB")
        if self.transforms is not None:
            img=self.transforms(img)
        caplen=len(self.data["CAPTIONS"][i])
        pad_caps=[self.vocab["<pad>"]] * (self.max_len + 2 - caplen)
        caption=torch.LongTensor(self.data["CAPTIONS"][i] + pad_caps)
        #返回结果是图片地址 ****.jpg， captions[1,2,3,4,5,100,100,100], 实际长度
        return img, caption, caplen
    def __len__(self):
        return self.dataset_size

    
"""
# 4  利用数据集类，通过dataloader构建产生批次训练、验证和测试的数据
# """
def mktrainval(data_dir, vocab_path, batch_size, workers=4):
    """data_dir路径为./data/datasets"""
    train_tx=transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tx=transforms.Compose([
        transforms.Resize(256),  #裁剪短边到256像素，保持宽高比不变
        transforms.CenterCrop(224), #从中心裁剪
        transforms.ToTensor(),    #形状自动转换为(C,H,W)值范围(0,1)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set=ImageTextDataset(os.path.join(data_dir,"train_data.json"), 
                               vocab_path, "train", transforms=train_tx)
    val_set=ImageTextDataset(os.path.join(data_dir,"val_data.json"), 
                               vocab_path, "val", transforms=val_tx)
    test_set=ImageTextDataset(os.path.join(data_dir,"test_data.json"), 
                               vocab_path, "test", transforms=val_tx)
    #pin_memory=True，将加载的数据锁页到GPU可直接访问的内存，加速cpu-gpu数据传输
    train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                            num_workers=workers, pin_memory=True, drop_last=True)
    valid_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=workers, pin_memory=True, drop_last=False)
    test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=workers, pin_memory=True, drop_last=False)
    
    """pin_memory 用法：
    现代操作系统使用虚拟内存机制， 物理内存（RAM）不足时， 将不常用的内存页交换到硬盘，需要时再交换回内存，
    使用pin_memory=true时候固定在内存上，加快传输
    
    """
    return train_loader, valid_loader, test_loader

"""图像表示提取器,使用在ImageNet预训练过的两个分类模型ResNet-152和VGG19作为图像表示提取器，需要修改最后一个全连接层"""
class ImageRepExtractor(nn.Module):
    def __init__(self, embed_size, pretrained_model="resnet152", finetuned=True):
        """
        embed_size 对应表示维度
        """
        super(ImageRepExtractor, self).__init__()
        if pretrained_model =="resnet152":
            net=torchvision.models.resnet152(weights=ResNet152_Weights.DEFAULT)
            for param in net.parameters():
                param.requires_grad=finetuned
            #finetuned：如果True，参数需要梯度，可以微调，如果false冻结参数，只作为特征提取器
            #更改最后一层(fc层)
            net.fc=nn.Linear(net.fc.in_features, embed_size)
            nn.init.xavier_uniform_(net.fc.weight)
        elif pretrained_model == "vgg19":
            net=torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)
            for param in net.parameters():
                param.requires_grad=finetuned
                #更改最后一层(fc层)
            net.classifier[6]=nn.Linear(net.classifier[6].in_features, embed_size)
            nn.init.xavier_uniform_(net.classifier[6].weight)
        else:
            raise ValueError("Unknown image model " + pretrained_model)
        self.net=net
    def forward(self, x): 
        output=self.net(x)
        output=nn.functional.normalize(output)   #对输出进行L2归一化，将输出向量的模长变为1
        return  output
            
"""文本表示提取器，使用GRU模型作为文本表示提取器，输入层为词嵌入形式，文本表示为最后一个词对应的隐藏层输出"""
class TextRepExtractor(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers):
        """
        word_dim: 词嵌入维度
        embed_size: 对应的表示维度，也是RNN隐藏层维度
        num_layers:RNN隐藏层数
        """
        super(TextRepExtractor, self).__init__()
        self.embed_size=embed_size
        self.embed=nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.rnn=nn.GRU(word_dim, embed_size,num_layers=num_layers, batch_first=True)
        #RNN默认已初始化，这里只需要初始化词嵌入矩阵
        self.embed.weight.data.uniform_(-0.1,0.1)
    def forward(self,x, lengths):
        x=self.embed(x)
        #压缩掉填充值
        packed=pack_padded_sequence(x, lengths=lengths, batch_first=True) #打包填充序列，只计算实际有效的部分
        #执行GRU的前馈过程会返回两个变量，第一个变量是每个时间步最后一个层输出，
        # 第二个变量hidden是最后一个时间步对应的所有隐藏层输出
        _, hidden=self.rnn(packed)   
        #最后一个词的最后一个隐藏层的输出为hidden[-1]作为文本表示
        out=nn.functional.normalize(hidden[-1])
        return out
    

"""VSE++模型：利用图像表示和文本表示提取器对成对的图像和文本数据输出表示
首先，按照文本的长短对数据进行排序
为了评测模型时能够对齐图像和文本数据，还需要恢复数据原始的输入顺序
"""

class VSEPP(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, image_model, finetuned=True):
        """
        image_model:图像表示提取器,ResNet-152 或VGG19
        finetuned:是否微调图像表示提取器的参数
        """
        super(VSEPP, self).__init__()
        self.image_extractor=ImageRepExtractor(embed_size, image_model,finetuned)
        self.text_extractor=TextRepExtractor(vocab_size, word_dim, embed_size, num_layers)
    def forward(self, images, captions, caplens):
        #按照captions的长短排序， 并对照调整image顺序
        sorted_cap_lens, sorted_cap_indices=torch.sort(caplens, dim=0, descending=True)#True表示descending降序 
        """
        sorted_cap_lens: 排序后的长度值
        sorted_cap_indices:排序后的索引，原位置到新位置的映射
        example:
            caplens=[3,5,2,4]
            sorted_cap_lens=[5,4,3,2]
            sorted_cap_indices=[1,3,0,2]
        """
        images=images[sorted_cap_indices]
        captions=captions[sorted_cap_indices]
        caplens=sorted_cap_lens
        image_code=self.image_extractor(images)
        #为了高效使用pack_padded_sequence，要求输入按照长度降序排列
        text_code=self.text_extractor(captions, caplens)
        if not self.training:
            #如果当前模型不在训练模式下，需要恢复原始输入数据的顺序，这是因为在评估或测试阶段，保持输入数据与预期输出之间的关系非常重要， 也就是在验证/测试阶段，需要恢复原始输入顺序，以保持和标签的对应关系
            _, recover_indices=torch.sort(sorted_cap_indices)   #默认是升序
            image_code=image_code[recover_indices]
            text_code=text_code[recover_indices]
        return image_code, text_code
    
"""定义损失函数：
    VSE++模型采用了困难样本挖掘的triplet损失函数，困难样本挖掘分为离线和在线挖掘两种方式,离线挖掘是在训练开始或每一轮训练结束后，挖掘困难样本；在线挖掘是在每一个批数据里，挖掘困难样本
"""
class TripletNetLoss(nn.Module):
    def __init__(self, margin=0.2, hard_negative=False):
        """hard_negative表示是否使用困难样本策略
        margin:三元组损失中的间隔
        """
        super(TripletNetLoss, self).__init__()
        self.margin=margin
        self.hard_negative=hard_negative
    def forward(self, ie, te):
        """
        ie:图像表示，形状为batch_size, embed_dim
        te:文本表示，形状为batch_size, embed_dim
        """
        scores=ie.mm(te.t())
        diagonal=scores.diag().view(ie.size(0), 1)
        d1=diagonal.expand_as(scores)
        d2=diagonal.t().expand_as(scores)
        #图像为锚
        cost_i=(self.margin + scores - d1).clamp(min=0)
        #文本为锚
        cost_t=(self.margin + scores  - d2).clamp(min=0)
        #损失矩阵对角线上的值不参与计算
        mask=torch.eye(scores.size(0), dtype=torch.bool)  #torch.eye生成一个对角线全为1，其余部分全为0的二维张量
        I=mask
        if torch.cuda.is_available():
            I=I.to(scores.device)
        cost_i=cost_i.masked_fill_(I,0)
        cost_t=cost_t.masked_fill_(I,0)
        #寻找困难样本
        if self.hard_negative:
            cost_i=cost_i.max(1)[0]
            cost_t=cost_t.max(0)[0]
        return cost_i.sum() +cost_t.sum()
    
"""选用Adam优化算法更新模型参数，学习率采用分段衰减方法"""
def get_optimizer(model, config):
    params=filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.Adam(params=params, lr=config.learning_rate)
def adjust_learning_rate(optimizer, epoch, config):
    """每隔lr_update个轮次，学习率减小至当前二分之一"""
    lr=config.learning_rate * (0.5 ** (epoch // config.lr_update))
    lr=max(lr, config.min_learning_rate)
    # Pytorch优化器内部将参数按“参数组”组织，每个参数组可以有自己的学习率等超参数
    for param_group in optimizer.param_groups:
        param_group["lr"]=lr
"""
optimizer.state形状是一个字典， 键是参数的引用，值是该参数对应的优化器状态字典，
例如，对于Adam优化器， 每个参数的状态可能包含：
    {
    "step":100,
    "exp_avg": tensor(...)
    }


"""
def move_optimizer_state_to_device(optimizer, device):
    #有些情况下，optimizer state tensor会在CPU，统一搬到GPU
    for state in optimizer.state.values:
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k]=v.to(device)

"""评估指标"""
def evaluate(data_loader, model, batch_size, captions_per_image):
    #切换模型进入评估模式,关闭dropout、batch normalization、不需要进行梯度计算和反向传播
    model.eval()
    #image_codes, text_codes 用于存储图像和文本的嵌入表示(编码)， 初始值为None
    image_codes=None
    text_codes=None
    #获取模型当前使用的设备，确保输入数据被正确移动到相同的设备上进行编码
    device=next(model.parameters()).device   #获取模型第一个参数所在的设备
    """pytorch要求模型参数和输入数据必须在同一设备上才能进行计算"""
    N=len(data_loader.dataset)  #获取数据集中总样本的数量
    """dataloader内部持有一个dataset属性，指向它所包装的原始Dataset对象"""
    for i, (imgs, captions, caplens) in enumerate(data_loader): #遍历dataloader提供的数据批次
        with torch.no_grad():
            image_code,  text_code=model(imgs.to(device), captions.to(device), caplens.to(device))
            if image_codes is None:
                image_codes=np.zeros((N, image_code.size(1)))
                text_codes=np.zeros((N, text_code.size(1)))
                #将图文对应表示存到numpy数组中，之后在cpu上计算recall
            #计算当前批次在image_codes和text_codes中的起始和结束索引，并将当前批次的图像和文本嵌入表示从GPU移动到CPU并存储到相应的数组中
            st=i * batch_size
            ed=(i+1) * batch_size
            image_codes[st:ed]=image_code.data.cpu().numpy()
            text_codes[st:ed]=text_code.data.cpu().numpy()
        #模型切换回训练模式
    model.train()
    return calc_recall(image_codes, text_codes, captions_per_image)

"""为什么要使用全部样本的原因：
    如果只使用批次样本，相似度计算不完整可能会导致图像A的最佳匹配文本可能在另一个批次中，召回率会严重低估
"""

def calc_recall(image_codes, text_codes, captions_per_image):
    """
    以步长caption_per_images对图像编码进行采样，假设有100张图像，每张5个描述，则text有500个，但是只取第0，5，10...张图像， 相当于每张图像只取第一个代表
    """
    #之所以可以每隔固定数量取图片，是因为前面对图文数据对输入顺序进行了还原
    scores=np.dot(image_codes[::captions_per_image], text_codes.T)
    #以图检文：按行从大到小排序
    sorted_scores_indices=(-scores).argsort(axis=1) #argsort默认是按照升序进行排列，使用负号就是按照相似度降序排列
    (n_images, n_texts)=scores.shape
    ranks_i2t=np.zeros(n_images)
    #创建数组存储每张图像的以图检文排名
    for i in range(n_images):
        #一张图片对应captions_per_image条描述，找到排名最靠前的文本位置
        min_rank=1e10
        for j in range(i * captions_per_image, (i+1) * captions_per_image):
            #返回的是一张图像对应的captions_per_image条描述，找到排名最靠前的文本位置
            rank=list(sorted_scores_indices[i,:]).index(j)
            #找到第j个文本在该图像相似度排序中的位置(排名)
            if min_rank > rank:
                min_rank=rank
        ranks_i2t[i]=min_rank
    """以文检图，按列从大到小进行排序"""
    sorted_scores_indices=(-scores).argsort(axis=0)
    ranks_t2i=np.zeros(n_texts)
    for i in range(n_texts):
        rank=list(sorted_scores_indices[:,i]).index(i // captions_per_image)
        ranks_t2i[i]=rank
        """
        为什么以文检图时不需要进行最小值判断，因为一条文本描述只对应一张图像
        """
    #最靠前的位置小于k，即recall@K, 这里计算了k取1，5，10时的图文互检的recall
    r1_i2t=100.0 * len(np.where(ranks_i2t < 1 )[0]) / n_images
    r1_t2i=100.0 * len(np.where(ranks_t2i < 1 )[0]) / n_texts
    r5_i2t=100.0 * len(np.where(ranks_i2t < 5 )[0]) / n_images
    r5_t2i=100.0 * len(np.where(ranks_t2i < 5 )[0]) / n_texts
    r10_i2t=100.0 * len(np.where(ranks_i2t < 10 )[0]) / n_images
    r10_t2i=100.0 * len(np.where(ranks_t2i < 10 )[0]) / n_texts
    return r1_i2t,r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i

"""训练模型过程分为读取数据，前馈计算，计算损失，更新参数，选择模型五个阶段"""
def main():
    config=Namespace(
        captions_per_image=5,
        batch_size=32,
        word_dim=300,
        embed_size=1024,
        num_layers=1,
        image_model="resnet152",
        finetuned=True,
        learning_rate=0.00002,
        lr_update=15,
        min_learning_rate=0.000002,
        margin=0.2,
        hard_negative=True,
        num_epoches=45,
        grad_clip=2,
        evaluate_step=60, #每隔多少步在验证集上测试一次，每训练60个批次(steps)进行一次验证集评估
        checkpoint=None, #如果不为None，则利用该变量存储路径的模型继续训练
        best_checkpoint="./model/vsepp/best_flickr8k.ckpt",  #验证集上表现最优模型的路径
        last_checkpoint="./model/vsepp/last_flickr8k.ckpt"  #训练完成时的模型的路径
    )
    #设置GPU信息
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device=torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")


    #数据
    data_dir="./data/datasets"
    vocab_path="./data/datasets/vocab.json"
    train_loader, valid_loader, test_loader=mktrainval(data_dir,vocab_path, config.batch_size)
    #模型
    with open(vocab_path, "r") as f:
        vocab=json.load(f)
    #随机初始化或载入已训练的模型
    model=VSEPP(len(vocab), config.word_dim, config.embed_size, config.num_layers, 
                    config.image_model, config.finetuned)
    optimizer=get_optimizer(model,config)
    loss_fn=TripletNetLoss(config.margin, config.hard_negative)

    #恢复训练逻辑
    start_epoch=0
    start_step=0
    best_res=0.0
    ckpt_path=None
    if config.checkpoint is not None:
        ckpt_path=config.checkpoint
    elif os.path.exists(config.last_checkpoint):
        ckpt_path=config.last_checkpoint
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path, map_location= device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        move_optimizer_state_to_device(optimizer, device)
        start_epoch=ckpt.get("epoch", -1) +1
        start_step=ckpt.get("step", 0)
        best_res=ckpt.get("best_res", 0.0)
        print(f">>>Resumed from {ckpt_path} @ epoch={start_epoch}, ste={start_step}, best_res= {best_res}")
    else:
        print(f"No checkpoint from {ckpt_path}. Training from the scratch")
    model.to(device)
    model.train()
    #添加变量初始化
    last_recall_sum = 0
    last_r1_i2t = 0
    last_r1_t2i = 0
    last_r5_i2t = 0
    last_r5_t2i = 0
    last_r10_i2t = 0
    last_r10_t2i = 0
    print("开始训练")
    for epoch in range(start_epoch, config.num_epoches):
        adjust_learning_rate(optimizer, epoch, config)
        for i, (imgs,caps, caplens) in enumerate(train_loader):
            optimizer.zero_grad()
            # 1 读取数据至GPU
            imgs=imgs.to(device, non_blocking=True)
            caps=caps.to(device, non_blocking=True)
            # 2 前馈计算
            image_codes, text_codes=model(imgs, caps, caplens)
            # 3 计算损失
            loss=loss_fn(image_codes, text_codes)
            loss.backward()
            #  梯度截断
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            # 4 更新参数
            optimizer.step()
            if (i+1) % config.evaluate_step==0:
                last_r1_i2t, last_r1_t2i,last_r5_i2t, last_r5_t2i, last_r10_i2t,  last_r10_t2i=evaluate(valid_loader, model, config.batch_size, config.captions_per_image)
                last_recall_sum= last_r1_i2t + last_r1_t2i + last_r5_i2t +last_r5_t2i
                + last_r10_i2t + last_r10_t2i
                state={
                    "epoch": epoch,
                    "step": i,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_res": best_res
                    }
                torch.save(state, config.last_checkpoint)
                if last_recall_sum > best_res:
                    best_res=last_recall_sum
                    state["best_res"]=best_res
                    torch.save(state, config.best_checkpoint)

                print(
                    "epoch: %d, step: %d, loss: %.4f | "
                    "I2T R@1: %.2f, T2I R@1: %.2f | "
                    "I2T R@5: %.2f, T2I R@5: %.2f | "
                    "I2T R@10: %.2f, T2I R@10: %.2f | "
                    "recall_sum: %.2f best: %.2f" %
                    (epoch, i + 1, loss.item(),
                     last_r1_i2t,last_r1_t2i, last_r5_i2t, last_r5_t2i, last_r10_i2t, last_r10_t2i,last_recall_sum, best_res),
                    flush=True
                )
    if  os.path.exists(config.best_checkpoint):
        ckpt=torch.load(config.best_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)


    r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i = evaluate(
        test_loader, model, config.batch_size, config.captions_per_image
    )
    print("\n=== Test Results (Best Model) ===")
    print("I2T R@1: %.2f, R@5: %.2f, R@10: %.2f" % (r1_i2t, r5_i2t, r10_i2t))
    print("T2I R@1: %.2f, R@5: %.2f, R@10: %.2f" % (r1_t2i, r5_t2i, r10_t2i))
if __name__=="__main__":
    main()










 










    




    
    
    














