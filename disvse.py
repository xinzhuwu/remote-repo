from argparse import Namespace 
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet152_Weights, VGG19_Weights
from PIL import Image
import os
import random
from collections import Counter, defaultdict

"""
1、下载数据集
2、整理数据集
"""
def create_dataset(dataset="flickr8k", captions_per_image=5, min_word_count=5, max_len=30):
    kerpathy_json_path = "data/dataset_flickr8k.json"
    image_folder = "data/Images"
    output_folder = "data/datasets"
    with open(kerpathy_json_path) as j:
        data = json.load(j)
    image_paths = defaultdict(list)
    image_captions = defaultdict(list)
    vocab = Counter()
    for img in data["images"]:
        split = img["split"]
        captions = []
        for c in img["sentences"]:
            if split != "test":
                vocab.update(c["tokens"])
            if len(c["tokens"]) <= max_len:
                captions.append(c["tokens"])
        if len(captions) == 0:
            continue
        path = os.path.join(image_folder, img["filename"])
        image_paths[split].append(path)
        image_captions[split].append(captions)

    words = [w for w in vocab.keys() if vocab[w] >= min_word_count]
    vocab = {k: v + 1 for v, k in enumerate(words)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = len(vocab)
    vocab["<start>"] = len(vocab)
    vocab["<end>"] = len(vocab)

    with open(os.path.join(output_folder, "vocab.json"), "w") as fw:
        json.dump(vocab, fw, indent=2, ensure_ascii=False)

    for split in image_paths:
        imgpaths = image_paths[split]
        imcaps = image_captions[split]
        enc_captions = []
        for i, path in enumerate(imgpaths):
            img = Image.open(path)
            if len(imcaps[i]) < captions_per_image:
                filled_num = captions_per_image - len(imcaps[i])
                captions = imcaps[i] + [random.choice(imcaps[i]) for _ in range(filled_num)]
            else:
                captions = random.sample(imcaps[i], k=captions_per_image)
            assert len(captions) == captions_per_image
            for j, c in enumerate(captions):
                enc_c = [vocab["<start>"]] + [vocab.get(word, vocab["<unk>"]) for word in c] + [vocab["<end>"]]
                enc_captions.append(enc_c)
        assert len(imgpaths) * captions_per_image == len(enc_captions)
        data = {"IMAGES": imgpaths, "CAPTIONS": enc_captions}
        with open(os.path.join(output_folder, split + "_data.json"), "w") as fw:
            json.dump(data, fw)


"""
3、定义数据集类
"""
class ImageTextDataset(Dataset):
    def __init__(self, dataset_path, vocab_path, split, captions_per_image=5, max_len=30, transforms=None):
        super(ImageTextDataset, self).__init__()
        self.split = split
        assert self.split in ["train", "test", "val"]
        self.cpi = captions_per_image
        self.max_len = max_len
        with open(dataset_path, "r") as f:
            self.data = json.load(f)
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.transforms = transforms
        self.dataset_size = len(self.data["CAPTIONS"])

    def __getitem__(self, i):
        img = Image.open(self.data["IMAGES"][i // self.cpi]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        caplen = len(self.data["CAPTIONS"][i])
        pad_caps = [self.vocab["<pad>"]] * (self.max_len + 2 - caplen)
        caption = torch.LongTensor(self.data["CAPTIONS"][i] + pad_caps)
        return img, caption, caplen

    def __len__(self):
        return self.dataset_size


"""
4. DataLoader
"""
def mktrainval(data_dir, vocab_path, batch_size, workers=4):
    train_tx = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = ImageTextDataset(os.path.join(data_dir, "train_data.json"), vocab_path, "train", transforms=train_tx)
    val_set = ImageTextDataset(os.path.join(data_dir, "val_data.json"), vocab_path, "val", transforms=val_tx)
    test_set = ImageTextDataset(os.path.join(data_dir, "test_data.json"), vocab_path, "test", transforms=val_tx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, valid_loader, test_loader


"""
图像表示提取器
"""
class ImageRepExtractor(nn.Module):
    def __init__(self, embed_size, pretrained_model="resnet152", finetuned=True):
        super(ImageRepExtractor, self).__init__()
        if pretrained_model == "resnet152":
            net = torchvision.models.resnet152(weights=ResNet152_Weights.DEFAULT)
            for param in net.parameters():
                param.requires_grad = finetuned
            net.fc = nn.Linear(net.fc.in_features, embed_size)
            nn.init.xavier_uniform_(net.fc.weight)
        elif pretrained_model == "vgg19":
            net = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)
            for param in net.parameters():
                param.requires_grad = finetuned
            net.classifier[6] = nn.Linear(net.classifier[6].in_features, embed_size)
            nn.init.xavier_uniform_(net.classifier[6].weight)
        else:
            raise ValueError("Unknown image model " + pretrained_model)
        self.net = net

    def forward(self, x):
        output = self.net(x)
        return output


"""
文本表示提取器
"""
class TextRepExtractor(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers):
        super(TextRepExtractor, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers=num_layers, batch_first=True)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        out = hidden[-1]  # [batch, embed_size]
        return out


"""
VSE++ 模型（支持 normalize 控制）
"""
class VSEPP(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, image_model, finetuned=True):
        super(VSEPP, self).__init__()
        self.image_extractor = ImageRepExtractor(embed_size, image_model, finetuned)
        self.text_extractor = TextRepExtractor(vocab_size, word_dim, embed_size, num_layers)

    def forward(self, images, captions, caplens, normalize=True):
        sorted_cap_lens, sorted_cap_indices = torch.sort(caplens, dim=0, descending=True)
        images = images[sorted_cap_indices]
        captions = captions[sorted_cap_indices]
        caplens = sorted_cap_lens

        image_code = self.image_extractor(images)
        text_code = self.text_extractor(captions, caplens)

        if not self.training:
            _, recover_indices = torch.sort(sorted_cap_indices)
            image_code = image_code[recover_indices]
            text_code = text_code[recover_indices]

        if normalize:
            image_code = nn.functional.normalize(image_code, dim=1)
            text_code = nn.functional.normalize(text_code, dim=1)

        return image_code, text_code


"""
判别器：判断图文对是否匹配
"""
class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, img_emb, txt_emb):
        x = torch.cat([img_emb, txt_emb], dim=1)
        return self.net(x).squeeze(-1)


"""
优化器 & 学习率调整
"""
def get_optimizer(model, config):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.Adam(params, lr=config.learning_rate)

def adjust_learning_rate(optimizer, epoch, config):
    lr = config.learning_rate * (0.5 ** (epoch // config.lr_update))
    lr = max(lr, config.min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


"""
评估指标（保持不变）
"""
def evaluate(data_loader, model, batch_size, captions_per_image):
    model.eval()
    device = next(model.parameters()).device
    N = len(data_loader.dataset)
    image_codes = None
    text_codes = None

    for i, (imgs, captions, caplens) in enumerate(data_loader):
        with torch.no_grad():
            image_code, text_code = model(imgs.to(device), captions.to(device), caplens.to(device))
            if image_codes is None:
                image_codes = np.zeros((N, image_code.size(1)))
                text_codes = np.zeros((N, text_code.size(1)))
            st = i * batch_size
            ed = min((i + 1) * batch_size, N)
            image_codes[st:ed] = image_code.cpu().numpy()
            text_codes[st:ed] = text_code.cpu().numpy()

    model.train()
    return calc_recall(image_codes, text_codes, captions_per_image)

def calc_recall(image_codes, text_codes, captions_per_image):
    scores = np.dot(image_codes[::captions_per_image], text_codes.T)
    n_images, n_texts = scores.shape

    # I2T
    sorted_scores_indices = (-scores).argsort(axis=1)
    ranks_i2t = np.zeros(n_images)
    for i in range(n_images):
        min_rank = 1e10
        for j in range(i * captions_per_image, (i + 1) * captions_per_image):
            rank = list(sorted_scores_indices[i, :]).index(j)
            min_rank = min(min_rank, rank)
        ranks_i2t[i] = min_rank

    # T2I
    sorted_scores_indices = (-scores).argsort(axis=0)
    ranks_t2i = np.zeros(n_texts)
    for i in range(n_texts):
        rank = list(sorted_scores_indices[:, i]).index(i // captions_per_image)
        ranks_t2i[i] = rank

    r1_i2t = 100.0 * np.sum(ranks_i2t < 1) / n_images
    r1_t2i = 100.0 * np.sum(ranks_t2i < 1) / n_texts
    r5_i2t = 100.0 * np.sum(ranks_i2t < 5) / n_images
    r5_t2i = 100.0 * np.sum(ranks_t2i < 5) / n_texts
    r10_i2t = 100.0 * np.sum(ranks_i2t < 10) / n_images
    r10_t2i = 100.0 * np.sum(ranks_t2i < 10) / n_texts
    return r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i


"""
主训练流程
"""
if __name__ == "__main__":
    config = Namespace(
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
        num_epoches=45,
        grad_clip=2,
        evaluate_step=60,
        checkpoint=None,
        best_checkpoint="./model/vsepp/best_flickr8k.ckpt",
        last_checkpoint="./model/vsepp/last_flickr8k.ckpt"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data/datasets/"
    vocab_path = "./data/datasets/vocab.json"

    # 创建目录
    os.makedirs(os.path.dirname(config.best_checkpoint), exist_ok=True)

    train_loader, valid_loader, test_loader = mktrainval(data_dir, vocab_path, config.batch_size)

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    start_epoch = 0
    if config.checkpoint is None:
        model = VSEPP(len(vocab), config.word_dim, config.embed_size, config.num_layers, config.image_model, config.finetuned)
    else:
        checkpoint = torch.load(config.checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model = checkpoint["model"]

    model = model.to(device)
    model.train()

    # === 新增：判别器 ===
    discriminator = Discriminator(config.embed_size).to(device)
    optimizer = get_optimizer(model, config)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)

    bce_loss = nn.BCEWithLogitsLoss()
    best_res = 0

    print("开始训练（对抗损失）...")
    for epoch in range(start_epoch, config.num_epoches):
        adjust_learning_rate(optimizer, epoch, config)
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            batch_size = imgs.size(0)

            # 编码（不归一化）
            image_codes, text_codes = model(imgs, caps, caplens, normalize=False)

            # 正样本（匹配）
            real_scores = discriminator(image_codes, text_codes)
            real_labels = torch.ones(batch_size, device=device)

            # 负样本（错位配对）
            text_mismatch = torch.cat([text_codes[1:], text_codes[:1]], dim=0)
            fake_scores = discriminator(image_codes, text_mismatch)
            fake_labels = torch.zeros(batch_size, device=device)

            # 判别器损失
            disc_real_loss = bce_loss(real_scores, real_labels)
            disc_fake_loss = bce_loss(fake_scores, fake_labels)
            disc_loss = disc_real_loss + disc_fake_loss

            disc_optimizer.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optimizer.step()

            # 编码器损失（欺骗判别器）
            fake_scores_gen = discriminator(image_codes, text_mismatch)
            gen_loss = bce_loss(fake_scores_gen, real_labels)

            optimizer.zero_grad()
            gen_loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # 评估
            if (i + 1) % config.evaluate_step == 0:
                r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i = evaluate(
                    valid_loader, model, config.batch_size, config.captions_per_image
                )
                recall_sum = r1_i2t + r1_t2i + r5_i2t + r5_t2i + r10_i2t + r10_t2i

                if recall_sum > best_res:
                    best_res = recall_sum
                    torch.save({
                        'epoch': epoch,
                        'step': i,
                        'model': model,
                        'optimizer': optimizer.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'disc_optimizer': disc_optimizer.state_dict()
                    }, config.best_checkpoint)

                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model': model,
                    'optimizer': optimizer.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'disc_optimizer': disc_optimizer.state_dict()
                }, config.last_checkpoint)

                print("Epoch: %d, Step: %d, Gen Loss: %.4f, Disc Loss: %.4f | "
                      "R@1 I2T: %.2f, T2I: %.2f" % 
                      (epoch, i+1, gen_loss.item(), disc_loss.item(), r1_i2t, r1_t2i))

    # 最终测试
    checkpoint = torch.load(config.best_checkpoint, map_location=device)
    model = checkpoint["model"]
    model = model.to(device)
    r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i = evaluate(
        test_loader, model, config.batch_size, config.captions_per_image
    )
    print("\n=== Test Results (Best Model) ===")
    print("I2T R@1: %.2f, R@5: %.2f, R@10: %.2f" % (r1_i2t, r5_i2t, r10_i2t))
    print("T2I R@1: %.2f, R@5: %.2f, R@10: %.2f" % (r1_t2i, r5_t2i, r10_t2i))