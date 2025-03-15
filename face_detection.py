#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple

import image_utils

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    检测图像中的人脸
    
    Args:
        image: 输入图像
        
    Returns:
        人脸矩形列表 [(x, y, w, h), ...]
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

def detect_eyes(image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    检测人脸区域中的眼睛
    
    Args:
        image: 输入图像
        face_rect: 人脸矩形 (x, y, w, h)
        
    Returns:
        眼睛矩形列表 [(x, y, w, h), ...]
    """
    # 提取人脸区域
    x, y, w, h = face_rect
    roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    # 加载眼睛检测器
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 调整眼睛坐标到原始图像坐标系
    return [(ex + x, ey + y, ew, eh) for ex, ey, ew, eh in eyes]

def mark_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
              detect_eyes_flag: bool = False) -> np.ndarray:
    """
    在图像上标记人脸
    
    Args:
        image: 输入图像
        faces: 人脸矩形列表 [(x, y, w, h), ...]
        detect_eyes_flag: 是否检测并标记眼睛
        
    Returns:
        标记后的图像
    """
    result = image.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制人脸矩形
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 添加标签
        cv2.putText(result, f'Face #{i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 如果需要，检测并标记眼睛
        if detect_eyes_flag:
            eyes = detect_eyes(image, (x, y, w, h))
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(result, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='人脸检测程序')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出图像路径')
    parser.add_argument('--eyes', action='store_true', help='是否检测眼睛')
    parser.add_argument('--display', action='store_true', help='是否显示结果')
    args = parser.parse_args()
    
    # 加载图像
    image = image_utils.load_image(args.image)
    
    # 检测人脸
    faces = detect_faces(image)
    
    # 标记人脸
    result = mark_faces(image, faces, args.eyes)
    
    # 显示结果
    if args.display:
        image_utils.display_image(result, "人脸检测结果")
    
    # 保存结果
    if args.output:
        image_utils.save_image(result, args.output)
        print(f"结果已保存到: {args.output}")
    
    # 打印检测到的人脸数量
    print(f"检测到 {len(faces)} 个人脸")

if __name__ == "__main__":
    main() 