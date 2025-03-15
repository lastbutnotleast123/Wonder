#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict

import image_utils

# 预训练模型的类别
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_ssd_model() -> cv2.dnn_Net:
    """
    加载预训练的SSD MobileNet模型
    
    Returns:
        加载的模型
    """
    # 检查模型文件是否存在，如果不存在则下载
    model_path = "models/ssd_mobilenet_v2_coco_2018_03_29.pb"
    config_path = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    
    # 创建模型目录
    os.makedirs("models", exist_ok=True)
    
    # 如果模型文件不存在，提示用户下载
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("模型文件不存在，请从以下链接下载：")
        print("https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API")
        print(f"并将文件放置在 {os.path.abspath('models')} 目录下")
        raise FileNotFoundError("模型文件不存在")
    
    # 加载模型
    model = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    
    return model

def detect_objects(image: np.ndarray, model: cv2.dnn_Net, 
                  confidence_threshold: float = 0.5) -> List[Dict]:
    """
    使用SSD模型检测图像中的物体
    
    Args:
        image: 输入图像
        model: 预训练的SSD模型
        confidence_threshold: 置信度阈值
        
    Returns:
        检测到的物体列表，每个物体包含类别、置信度和边界框
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 准备输入blob
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    
    # 设置模型输入
    model.setInput(blob)
    
    # 前向传播
    output = model.forward()
    
    # 解析检测结果
    detections = []
    for detection in output[0, 0]:
        # 提取置信度
        confidence = float(detection[2])
        
        # 过滤低置信度的检测结果
        if confidence > confidence_threshold:
            # 提取类别索引
            class_id = int(detection[1])
            
            # 提取边界框坐标
            box_x = int(detection[3] * width)
            box_y = int(detection[4] * height)
            box_width = int(detection[5] * width) - box_x
            box_height = int(detection[6] * height) - box_y
            
            # 添加到检测结果列表
            detections.append({
                'class_id': class_id,
                'class_name': COCO_CLASSES[class_id],
                'confidence': confidence,
                'box': (box_x, box_y, box_width, box_height)
            })
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测到的物体列表
        
    Returns:
        标记后的图像
    """
    result = image.copy()
    
    for detection in detections:
        # 提取信息
        class_name = detection['class_name']
        confidence = detection['confidence']
        box_x, box_y, box_width, box_height = detection['box']
        
        # 绘制边界框
        cv2.rectangle(result, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 2)
        
        # 准备标签文本
        label = f"{class_name}: {confidence:.2f}"
        
        # 绘制标签背景
        cv2.rectangle(result, (box_x, box_y - 20), (box_x + len(label) * 9, box_y), 
                     (0, 255, 0), -1)
        
        # 绘制标签文本
        cv2.putText(result, label, (box_x, box_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='物体检测程序')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出图像路径')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--display', action='store_true', help='是否显示结果')
    args = parser.parse_args()
    
    try:
        # 加载模型
        model = load_ssd_model()
        
        # 加载图像
        image = image_utils.load_image(args.image)
        
        # 检测物体
        detections = detect_objects(image, model, args.confidence)
        
        # 在图像上标记检测结果
        result = draw_detections(image, detections)
        
        # 显示结果
        if args.display:
            image_utils.display_image(result, "物体检测结果")
        
        # 保存结果
        if args.output:
            image_utils.save_image(result, args.output)
            print(f"结果已保存到: {args.output}")
        
        # 打印检测到的物体
        print(f"检测到 {len(detections)} 个物体:")
        for i, detection in enumerate(detections):
            print(f"{i+1}. {detection['class_name']} (置信度: {detection['confidence']:.2f})")
    
    except FileNotFoundError as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 