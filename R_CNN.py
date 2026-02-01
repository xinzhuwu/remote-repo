import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

class SelectiveSearch:
    """简化的选择性搜索实现"""
    def __init__(self):
        # 尝试创建选择性搜索对象
        try:
            self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        except:
            print("警告: OpenCV的selective search不可用，使用简化的区域生成方法")
            self.ss = None
    
    def extract_regions(self, image, strategy='quality'):
        """提取候选区域"""
        if self.ss is not None:
            self.ss.setBaseImage(image)
            
            if strategy == 'fast':
                self.ss.switchToSelectiveSearchFast()
            else:
                self.ss.switchToSelectiveSearchQuality()
            
            rects = self.ss.process()
            
            # 过滤掉太小的区域
            filtered_rects = []
            for x, y, w, h in rects:
                if w > 20 and h > 20:  # 最小尺寸阈值
                    filtered_rects.append((x, y, w, h))
            
            return filtered_rects[:200]  # 限制数量（为了速度）
        else:
            # 使用滑动窗口作为替代方案
            return self._sliding_window(image)
    
    def _sliding_window(self, image, window_size=(100, 100), stride=50):
        """滑动窗口生成候选区域"""
        h, w = image.shape[:2]
        regions = []
        
        for y in range(0, h - window_size[1], stride):
            for x in range(0, w - window_size[0], stride):
                regions.append((x, y, window_size[0], window_size[1]))
        
        # 添加不同尺度的窗口
        scales = [0.5, 0.75, 1.25, 1.5]
        for scale in scales:
            new_w = int(window_size[0] * scale)
            new_h = int(window_size[1] * scale)
            if new_w < w and new_h < h:
                for y in range(0, h - new_h, stride*2):
                    for x in range(0, w - new_w, stride*2):
                        regions.append((x, y, new_w, new_h))
        
        return regions[:200]  # 限制数量

class RCNN:
    """简化版R-CNN实现"""
    def __init__(self, num_classes=3, device='cpu'):
        self.device = device
        self.num_classes = num_classes  # 包含背景类
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 区域提取器
        self.selective_search = SelectiveSearch()
        
        # 初始化模型组件
        self.is_trained = False
        self.feature_extractor = None
        self.classifiers = None
        self.class_names = ['background']  # 背景类
        
    def build_feature_extractor(self):
        """构建特征提取网络"""
        # 使用预训练的VGG16（更现代的模型）
        model = models.vgg16(pretrained=True)
        
        # 移除最后的分类层，保留特征提取部分
        features = list(model.features.children())
        
        # 添加自定义层以获取特征向量
        self.feature_extractor = nn.Sequential(
            *features,
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        
        # 冻结预训练层
        for param in list(self.feature_extractor.children())[:-4]:
            param.requires_grad = False
        
        self.feature_extractor.to(self.device).eval()
        return self.feature_extractor
    
    def extract_features(self, image_pil):
        """提取图像特征"""
        if self.feature_extractor is None:
            self.build_feature_extractor()
        
        with torch.no_grad():
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            features = self.feature_extractor(image_tensor)
        return features.cpu().numpy().flatten()
    
    def create_dummy_data(self, image_size=(500, 500, 3)):
        """创建虚拟训练数据用于演示"""
        print("创建虚拟训练数据...")
        
        # 创建一些虚拟图像和标注
        num_images = 10
        images = []
        gt_boxes_list = []
        gt_labels_list = []
        
        # 定义类别
        self.class_names = ['background', 'person', 'car']
        self.num_classes = len(self.class_names)
        
        for i in range(num_images):
            # 创建随机图像
            image = np.random.randint(0, 255, image_size, dtype=np.uint8)
            images.append(image)
            
            # 创建虚拟标注
            num_objects = np.random.randint(1, 4)
            boxes = []
            labels = []
            
            for j in range(num_objects):
                # 随机生成边界框
                x = np.random.randint(0, image_size[1] - 100)
                y = np.random.randint(0, image_size[0] - 100)
                w = np.random.randint(50, 150)
                h = np.random.randint(50, 150)
                
                boxes.append((x, y, w, h))
                labels.append(np.random.randint(1, self.num_classes))  # 跳过背景类
            
            gt_boxes_list.append(boxes)
            gt_labels_list.append(labels)
        
        return images, gt_boxes_list, gt_labels_list
    
    def train(self, images=None, gt_boxes=None, gt_labels=None, iou_threshold=0.5):
        """训练R-CNN模型"""
        print("开始训练R-CNN模型...")
        
        # 如果没有提供数据，使用虚拟数据
        if images is None:
            images, gt_boxes, gt_labels = self.create_dummy_data()
        
        # 初始化存储结构
        positive_samples = [[] for _ in range(self.num_classes)]
        negative_samples = []
        
        total_regions = 0
        
        for idx, (image, gt_boxes_img, gt_labels_img) in enumerate(zip(images, gt_boxes, gt_labels)):
            print(f"处理图像 {idx+1}/{len(images)}")
            
            # 提取候选区域
            candidate_regions = self.selective_search.extract_regions(image)
            total_regions += len(candidate_regions)
            
            for region in candidate_regions:
                x, y, w, h = region
                
                # 提取区域图像
                if y + h > image.shape[0] or x + w > image.shape[1]:
                    continue
                    
                region_img = image[y:y+h, x:x+w]
                if region_img.size == 0:
                    continue
                
                # 转换为PIL图像
                region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
                
                # 提取特征
                features = self.extract_features(region_pil)
                
                # 计算与所有真实框的IoU
                max_iou = 0
                best_label = 0  # 0表示背景
                
                for gt_box, gt_label in zip(gt_boxes_img, gt_labels_img):
                    iou = self._calculate_iou(region, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        if iou > iou_threshold:
                            best_label = gt_label
                
                # 分配正负样本
                if best_label > 0:  # 正样本
                    positive_samples[best_label].append(features)
                else:  # 负样本（背景）
                    negative_samples.append(features)
        
        print(f"处理了 {total_regions} 个候选区域")
        print(f"正样本分布: {[len(samples) for samples in positive_samples]}")
        print(f"负样本数量: {len(negative_samples)}")
        
        # 训练SVM分类器
        print("\n训练分类器...")
        self.classifiers = [None] * self.num_classes
        
        for class_id in range(1, self.num_classes):  # 跳过背景类
            if len(positive_samples[class_id]) > 0:
                # 准备训练数据
                num_pos = len(positive_samples[class_id])
                num_neg = min(len(negative_samples), num_pos * 3)
                
                if num_neg > 0:
                    X_pos = np.array(positive_samples[class_id])
                    X_neg = np.array(negative_samples[:num_neg])
                    
                    X = np.vstack([X_pos, X_neg])
                    y = np.hstack([np.ones(num_pos), np.zeros(num_neg)])
                    
                    # 训练SVM
                    svm = LinearSVC(max_iter=10000)
                    svm.fit(X, y)
                    self.classifiers[class_id] = svm
                    
                    print(f"训练类别 {self.class_names[class_id]} 的SVM，准确率: {svm.score(X, y):.3f}")
                else:
                    print(f"类别 {self.class_names[class_id]} 没有足够的负样本")
            else:
                print(f"类别 {self.class_names[class_id]} 没有正样本")
        
        self.is_trained = True
        print("训练完成！")
    
    def predict(self, image, confidence_threshold=0.5):
        """预测图像中的物体"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！请先调用train()方法")
        
        # 提取候选区域
        candidate_regions = self.selective_search.extract_regions(image)
        
        detections = []
        
        print(f"处理 {len(candidate_regions)} 个候选区域...")
        
        for i, region in enumerate(candidate_regions):
            x, y, w, h = region
            
            # 提取区域图像
            if y + h > image.shape[0] or x + w > image.shape[1]:
                continue
                
            region_img = image[y:y+h, x:x+w]
            if region_img.size == 0:
                continue
            
            # 转换为PIL图像
            region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            
            # 提取特征
            features = self.extract_features(region_pil).reshape(1, -1)
            
            # 对每个类别进行分类
            for class_id in range(1, self.num_classes):
                classifier = self.classifiers[class_id]
                if classifier is not None:
                    try:
                        # 使用decision_function获取分数
                        score = classifier.decision_function(features)[0]
                        
                        if score > confidence_threshold:
                            detections.append({
                                'bbox': (x, y, w, h),
                                'class_id': class_id,
                                'score': score,
                                'class_name': self.class_names[class_id]
                            })
                    except:
                        continue
        
        # 非极大值抑制
        detections = self._non_maximum_suppression(detections)
        
        return detections
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 转换为中心点+宽高表示
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # 计算交集坐标
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        # 计算交集面积
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _non_maximum_suppression(self, detections, iou_threshold=0.3):
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 按分数排序
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        while detections:
            # 取分数最高的检测
            best = detections.pop(0)
            keep.append(best)
            
            # 移除与当前检测框重叠度高的框
            to_remove = []
            for i, det in enumerate(detections):
                if det['class_id'] == best['class_id']:
                    iou = self._calculate_iou(best['bbox'], det['bbox'])
                    if iou > iou_threshold:
                        to_remove.append(i)
            
            # 从后往前移除，避免索引错乱
            for i in sorted(to_remove, reverse=True):
                detections.pop(i)
        
        return keep

def visualize_detections(image, detections, title="检测结果"):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # 转换BGR到RGB用于显示
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    ax.imshow(display_image)
    ax.set_title(title)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
    
    for det in detections:
        x, y, w, h = det['bbox']
        class_id = det['class_id']
        class_name = det.get('class_name', f'class_{class_id}')
        score = det['score']
        
        # 选择颜色
        color = colors[class_id % len(colors)]
        
        # 绘制边界框
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        label = f"{class_name}: {score:.2f}"
        ax.text(x, y-5, label, color=color, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("=" * 50)
    print("R-CNN 目标检测演示")
    print("=" * 50)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建R-CNN实例
    rcnn = RCNN(num_classes=3, device=device)
    
    # 步骤1: 训练模型（使用虚拟数据）
    print("\n步骤1: 训练模型")
    rcnn.train()
    
    # 步骤2: 测试模型
    print("\n步骤2: 测试模型")
    
    # 创建测试图像
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 在测试图像上画一些形状作为"物体"
    cv2.rectangle(test_image, (100, 100), (200, 250), (255, 0, 0), -1)  # 蓝色矩形
    cv2.circle(test_image, (400, 200), 60, (0, 255, 0), -1)  # 绿色圆形
    cv2.rectangle(test_image, (300, 300), (450, 380), (0, 0, 255), -1)  # 红色矩形
    
    # 添加一些噪声
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # 进行预测
    try:
        detections = rcnn.predict(test_image, confidence_threshold=0.3)
        print(f"检测到 {len(detections)} 个物体")
        
        # 显示检测结果
        for i, det in enumerate(detections):
            print(f"检测 {i+1}: {det['class_name']} (分数: {det['score']:.3f}), "
                  f"边界框: {det['bbox']}")
        
        # 可视化
        visualize_detections(test_image, detections, "R-CNN检测结果")
        
    except ValueError as e:
        print(f"错误: {e}")

def real_image_example():
    """使用真实图像的例子"""
    # 你可以用这个函数测试真实图像
    # 注意：需要安装opencv-contrib-python以获得selective search功能
    
    print("\n使用真实图像的示例:")
    
    # 下载测试图像
    import urllib.request
    test_image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/blox.jpg"
    
    try:
        # 下载图像
        urllib.request.urlretrieve(test_image_url, "test_image.jpg")
        
        # 加载图像
        image = cv2.imread("test_image.jpg")
        
        if image is not None:
            # 创建R-CNN实例
            rcnn = RCNN(num_classes=3, device='cpu')
            
            # 训练（使用虚拟数据）
            rcnn.train()
            
            # 预测
            detections = rcnn.predict(image, confidence_threshold=0.5)
            
            # 可视化
            visualize_detections(image, detections, "真实图像检测结果")
            
            # 清理
            if os.path.exists("test_image.jpg"):
                os.remove("test_image.jpg")
                
    except Exception as e:
        print(f"处理真实图像时出错: {e}")
        print("请确保已安装opencv-contrib-python: pip install opencv-contrib-python")

if __name__ == "__main__":
    # 运行演示
    main()
    
    # 如果你想尝试真实图像，取消下面的注释
    # real_image_example()