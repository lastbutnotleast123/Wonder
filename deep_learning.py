#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import time
from typing import List, Tuple, Dict, Optional, Union

import image_utils

class DeepModel:
    """OpenCV深度学习模型基类"""
    
    def __init__(self, model_path: str, config_path: str, 
                framework: str = "tensorflow"):
        """
        初始化深度学习模型
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            framework: 深度学习框架，可选值为 'tensorflow', 'caffe', 'darknet', 'torch'
        """
        self.model_path = model_path
        self.config_path = config_path
        self.framework = framework.lower()
        self.net = None
        self.input_size = (300, 300)  # 默认输入大小
        self.scale_factor = 1.0  # 默认缩放因子
        self.mean = (0, 0, 0)  # 默认均值
        self.swap_rb = True  # 默认交换RB通道
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 如果配置文件路径不为空，检查是否存在
        if config_path and not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self) -> None:
        """加载深度学习模型"""
        if self.framework == "tensorflow":
            if self.config_path:
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
            else:
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
        
        elif self.framework == "caffe":
            self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        
        elif self.framework == "darknet":
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
        
        elif self.framework == "torch":
            self.net = cv2.dnn.readNetFromTorch(self.model_path)
        
        else:
            raise ValueError(f"不支持的深度学习框架: {self.framework}")
        
        print(f"已加载 {self.framework} 模型: {os.path.basename(self.model_path)}")
    
    def set_backend_target(self, backend: Optional[int] = None, 
                          target: Optional[int] = None) -> None:
        """
        设置计算后端和目标
        
        Args:
            backend: 计算后端，如 cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA
            target: 计算目标，如 cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA
        """
        if backend is not None:
            self.net.setPreferableBackend(backend)
        
        if target is not None:
            self.net.setPreferableTarget(target)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的输入blob
        """
        # 创建输入blob
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=self.scale_factor,
            size=self.input_size,
            mean=self.mean,
            swapRB=self.swap_rb,
            crop=False
        )
        
        return blob
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            image: 输入图像
            
        Returns:
            网络输出
        """
        # 预处理图像
        blob = self.preprocess(image)
        
        # 设置网络输入
        self.net.setInput(blob)
        
        # 获取输出层名称（如果可能）
        try:
            output_layers = self.net.getUnconnectedOutLayersNames()
            # 进行前向传播
            outputs = self.net.forward(output_layers)
        except:
            # 如果无法获取输出层名称，直接前向传播
            outputs = self.net.forward()
        
        return outputs

class ObjectDetector(DeepModel):
    """物体检测模型类"""
    
    def __init__(self, model_path: str, config_path: str, 
                framework: str = "tensorflow",
                classes_file: Optional[str] = None):
        """
        初始化物体检测模型
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            framework: 深度学习框架
            classes_file: 类别文件路径
        """
        super().__init__(model_path, config_path, framework)
        
        # 设置默认参数
        self.input_size = (300, 300)
        self.scale_factor = 1.0
        self.mean = (127.5, 127.5, 127.5)
        self.confidence_threshold = 0.5
        
        # 加载类别
        self.classes = []
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"已加载 {len(self.classes)} 个类别")
    
    def detect(self, image: np.ndarray, 
              confidence_threshold: float = 0.5) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            
        Returns:
            检测结果列表 [{class_id, class_name, confidence, box}, ...]
        """
        # 保存输入图像尺寸
        height, width = image.shape[:2]
        
        # 进行推理
        outputs = self.infer(image)
        
        # 解析检测结果
        detections = []
        
        # 对于SSD模型（TensorFlow Object Detection API）
        if isinstance(outputs, np.ndarray) and len(outputs.shape) == 4:
            # 典型的TensorFlow检测输出
            for detection in outputs[0, 0]:
                # 确保有足够的元素
                if len(detection) >= 7:
                    # 提取置信度
                    confidence = float(detection[2])
                    
                    # 过滤低置信度的检测结果
                    if confidence > confidence_threshold:
                        # 提取类别索引
                        class_id = int(detection[1])
                        
                        # 提取边界框坐标（归一化）
                        box_x = int(detection[3] * width)
                        box_y = int(detection[4] * height)
                        box_width = int(detection[5] * width) - box_x
                        box_height = int(detection[6] * height) - box_y
                        
                        # 添加到检测结果列表
                        class_name = self.classes[class_id] if 0 <= class_id < len(self.classes) else f"类别{class_id}"
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'box': (box_x, box_y, box_width, box_height)
                        })
        
        # 对于YOLO模型
        elif isinstance(outputs, list) and len(outputs) >= 1:
            boxes = []
            confidences = []
            class_ids = []
            
            # 解析每个输出层
            for output in outputs:
                # 对每个检测
                for detection in output:
                    # YOLO格式通常是：[x, y, width, height, conf, class_score1, class_score2, ...]
                    if len(detection) > 5:
                        # 获取置信度
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id] * detection[4]  # 类别置信度 * 目标置信度
                        
                        if confidence > confidence_threshold:
                            # YOLO返回的是中心坐标和宽高
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            # 转换为左上角坐标
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append((x, y, w, h))
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            # 应用非极大值抑制
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            
            if len(indices) > 0:
                # OpenCV 4.x 返回的indices格式不同
                try:
                    for i in indices:
                        i = i[0] if isinstance(i, (list, tuple)) else i
                        box = boxes[i]
                        class_id = class_ids[i]
                        class_name = self.classes[class_id] if 0 <= class_id < len(self.classes) else f"类别{class_id}"
                        
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidences[i],
                            'box': box
                        })
                except:
                    # 对于更新版本的OpenCV
                    for i in indices:
                        box = boxes[i]
                        class_id = class_ids[i]
                        class_name = self.classes[class_id] if 0 <= class_id < len(self.classes) else f"类别{class_id}"
                        
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidences[i],
                            'box': box
                        })
        
        return detections

class ImageClassifier(DeepModel):
    """图像分类模型类"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, 
                framework: str = "tensorflow",
                classes_file: Optional[str] = None):
        """
        初始化图像分类模型
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            framework: 深度学习框架
            classes_file: 类别文件路径
        """
        super().__init__(model_path, config_path, framework)
        
        # 设置默认参数
        self.input_size = (224, 224)  # ImageNet标准输入大小
        self.scale_factor = 1/255.0
        self.mean = (0.485, 0.456, 0.406)  # ImageNet标准均值
        self.std = (0.229, 0.224, 0.225)   # ImageNet标准差
        
        # 加载类别
        self.classes = []
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"已加载 {len(self.classes)} 个类别")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的输入blob
        """
        # 创建输入blob
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=self.scale_factor,
            size=self.input_size,
            mean=self.mean,
            swapRB=True,
            crop=False
        )
        
        # 如果使用标准差归一化
        if hasattr(self, 'std'):
            blob /= np.array(self.std).reshape(1, 3, 1, 1)
        
        return blob
    
    def classify(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        对图像进行分类
        
        Args:
            image: 输入图像
            top_k: 返回的最可能类别数量
            
        Returns:
            分类结果列表 [(class_id, class_name, probability), ...]
        """
        # 进行推理
        outputs = self.infer(image)
        
        # 解析分类结果
        if isinstance(outputs, np.ndarray):
            # 获取概率
            probabilities = outputs[0] if outputs.ndim > 1 else outputs
            
            # 获取top_k个最可能的类别
            if top_k > len(probabilities):
                top_k = len(probabilities)
            
            indices = np.argsort(probabilities)[-top_k:][::-1]
            
            # 提取类别和概率
            results = []
            for i in indices:
                class_name = self.classes[i] if i < len(self.classes) else f"类别{i}"
                results.append((int(i), class_name, float(probabilities[i])))
            
            return results
        
        # 处理其他格式的输出
        else:
            print("未知的输出格式，无法解析")
            return []

class ImageSegmenter(DeepModel):
    """图像分割模型类"""
    
    def __init__(self, model_path: str, config_path: str, 
                framework: str = "tensorflow",
                classes_file: Optional[str] = None):
        """
        初始化图像分割模型
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            framework: 深度学习框架
            classes_file: 类别文件路径
        """
        super().__init__(model_path, config_path, framework)
        
        # 设置默认参数
        self.input_size = (513, 513)  # DeepLab默认输入大小
        self.scale_factor = 1.0
        self.mean = (127.5, 127.5, 127.5)
        
        # 颜色映射
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        # 确保背景是黑色
        self.colors[0] = [0, 0, 0]
        
        # 加载类别
        self.classes = []
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"已加载 {len(self.classes)} 个类别")
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对图像进行语义分割
        
        Args:
            image: 输入图像
            
        Returns:
            (分割掩码, 可视化结果)
        """
        # 保存原始图像尺寸
        original_h, original_w = image.shape[:2]
        
        # 进行推理
        outputs = self.infer(image)
        
        # 处理输出
        if isinstance(outputs, np.ndarray):
            # 预测的分类掩码，形状为 [1, num_classes, height, width]
            segmentation = outputs[0]
            
            # 获取每个像素的类别索引
            mask = np.argmax(segmentation, axis=0)
            
            # 调整掩码大小以匹配原始图像
            mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), 
                             interpolation=cv2.INTER_NEAREST)
            
            # 创建可视化结果
            visualization = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            for i in range(len(self.colors)):
                visualization[mask == i] = self.colors[i]
            
            # 半透明叠加
            alpha = 0.6
            vis_image = cv2.addWeighted(image, 1-alpha, visualization, alpha, 0)
            
            return mask, vis_image
        
        # 处理其他格式的输出
        else:
            print("未知的输出格式，无法解析")
            return np.zeros((original_h, original_w), dtype=np.uint8), image.copy()

def download_model(url: str, save_path: str) -> bool:
    """
    下载模型文件
    
    Args:
        url: 模型文件URL
        save_path: 保存路径
        
    Returns:
        是否下载成功
    """
    try:
        import urllib.request
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 下载文件
        print(f"下载模型文件: {url}")
        urllib.request.urlretrieve(url, save_path)
        
        return os.path.exists(save_path)
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OpenCV深度学习模块")
    parser.add_argument("--mode", choices=["classify", "detect", "segment"], required=True,
                       help="操作模式：图像分类(classify)、物体检测(detect)或图像分割(segment)")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--classes", help="类别文件路径")
    parser.add_argument("--framework", choices=["tensorflow", "caffe", "darknet", "torch"],
                       default="tensorflow", help="深度学习框架")
    parser.add_argument("--output", help="输出图像路径")
    parser.add_argument("--display", action="store_true", help="是否显示结果")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU加速")
    args = parser.parse_args()
    
    # 加载图像
    image = image_utils.load_image(args.image)
    
    # 检查模型和配置文件是否存在
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    if args.config and not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return
    
    # 根据模式创建相应的模型
    try:
        # 设置GPU加速
        backend = cv2.dnn.DNN_BACKEND_CUDA if args.gpu else cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CUDA if args.gpu else cv2.dnn.DNN_TARGET_CPU
        
        if args.mode == "classify":
            # 图像分类
            model = ImageClassifier(args.model, args.config, args.framework, args.classes)
            model.set_backend_target(backend, target)
            
            # 开始计时
            start_time = time.time()
            
            # 分类图像
            results = model.classify(image)
            
            # 计算用时
            elapsed_time = time.time() - start_time
            
            # 处理结果
            if results:
                # 在图像上显示结果
                result_image = image.copy()
                for i, (class_id, class_name, probability) in enumerate(results):
                    text = f"{class_name}: {probability:.2f}"
                    cv2.putText(result_image, text, (10, 30 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 显示结果
                if args.display:
                    image_utils.display_image(result_image, "分类结果")
                
                # 保存结果
                if args.output:
                    image_utils.save_image(result_image, args.output)
                    print(f"结果已保存到: {args.output}")
                
                # 打印分类结果
                print(f"分类完成，用时 {elapsed_time:.2f} 秒")
                for i, (class_id, class_name, probability) in enumerate(results):
                    print(f"#{i+1}: {class_name} (概率: {probability:.4f})")
            else:
                print("未能获得分类结果")
        
        elif args.mode == "detect":
            # 物体检测
            model = ObjectDetector(args.model, args.config, args.framework, args.classes)
            model.set_backend_target(backend, target)
            
            # 开始计时
            start_time = time.time()
            
            # 检测物体
            detections = model.detect(image)
            
            # 计算用时
            elapsed_time = time.time() - start_time
            
            # 处理结果
            if detections:
                # 在图像上绘制检测结果
                result_image = image.copy()
                for detection in detections:
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    x, y, w, h = detection['box']
                    
                    # 绘制边界框
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(result_image, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示结果
                if args.display:
                    image_utils.display_image(result_image, "检测结果")
                
                # 保存结果
                if args.output:
                    image_utils.save_image(result_image, args.output)
                    print(f"结果已保存到: {args.output}")
                
                # 打印检测结果
                print(f"检测完成，共检测到 {len(detections)} 个物体，用时 {elapsed_time:.2f} 秒")
                for i, detection in enumerate(detections):
                    print(f"#{i+1}: {detection['class_name']} (置信度: {detection['confidence']:.4f})")
            else:
                print("未检测到物体")
        
        elif args.mode == "segment":
            # 图像分割
            model = ImageSegmenter(args.model, args.config, args.framework, args.classes)
            model.set_backend_target(backend, target)
            
            # 开始计时
            start_time = time.time()
            
            # 分割图像
            mask, vis_image = model.segment(image)
            
            # 计算用时
            elapsed_time = time.time() - start_time
            
            # 显示结果
            if args.display:
                image_utils.display_image(vis_image, "分割结果")
            
            # 保存结果
            if args.output:
                image_utils.save_image(vis_image, args.output)
                print(f"结果已保存到: {args.output}")
            
            # 打印统计信息
            print(f"分割完成，用时 {elapsed_time:.2f} 秒")
            print(f"分割掩码形状: {mask.shape}")
            
            # 统计各类别像素数量
            for i in range(np.max(mask) + 1):
                count = np.sum(mask == i)
                if count > 0:
                    class_name = model.classes[i] if i < len(model.classes) else f"类别{i}"
                    percentage = count / (mask.shape[0] * mask.shape[1]) * 100
                    print(f"类别 {i} ({class_name}): {count} 像素 ({percentage:.2f}%)")
    
    except Exception as e:
        import traceback
        print(f"处理过程中出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 