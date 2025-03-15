#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from typing import Tuple

import image_utils

def detect_edges_canny(image: np.ndarray, 
                      threshold1: int = 100, 
                      threshold2: int = 200) -> np.ndarray:
    """
    使用Canny算法检测图像边缘
    
    Args:
        image: 输入图像
        threshold1: 第一阈值
        threshold2: 第二阈值
        
    Returns:
        边缘检测结果图像
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 应用Canny边缘检测
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    return edges

def detect_edges_sobel(image: np.ndarray) -> np.ndarray:
    """
    使用Sobel算子检测图像边缘
    
    Args:
        image: 输入图像
        
    Returns:
        边缘检测结果图像
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # 合并梯度
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

def detect_edges_laplacian(image: np.ndarray) -> np.ndarray:
    """
    使用拉普拉斯算子检测图像边缘
    
    Args:
        image: 输入图像
        
    Returns:
        边缘检测结果图像
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # 转换为8位无符号整数
    laplacian = cv2.convertScaleAbs(laplacian)
    
    return laplacian

def overlay_edges(image: np.ndarray, edges: np.ndarray, 
                 color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    将边缘叠加到原始图像上
    
    Args:
        image: 原始图像
        edges: 边缘图像
        color: 边缘颜色 (B,G,R)
        
    Returns:
        叠加后的图像
    """
    result = image.copy()
    
    # 创建彩色边缘图像
    color_edges = np.zeros_like(image)
    for i in range(3):
        color_edges[:, :, i] = edges * (color[i] / 255.0)
    
    # 将边缘叠加到原始图像上
    alpha = 0.7
    beta = 1.0 - alpha
    result = cv2.addWeighted(result, alpha, color_edges.astype(np.uint8), beta, 0)
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='边缘检测程序')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出图像路径')
    parser.add_argument('--method', default='canny', choices=['canny', 'sobel', 'laplacian'], 
                       help='边缘检测方法')
    parser.add_argument('--overlay', action='store_true', help='是否将边缘叠加到原始图像上')
    parser.add_argument('--display', action='store_true', help='是否显示结果')
    parser.add_argument('--t1', type=int, default=100, help='Canny算法的第一阈值')
    parser.add_argument('--t2', type=int, default=200, help='Canny算法的第二阈值')
    args = parser.parse_args()
    
    # 加载图像
    image = image_utils.load_image(args.image)
    
    # 根据选择的方法检测边缘
    if args.method == 'canny':
        edges = detect_edges_canny(image, args.t1, args.t2)
    elif args.method == 'sobel':
        edges = detect_edges_sobel(image)
    elif args.method == 'laplacian':
        edges = detect_edges_laplacian(image)
    
    # 如果需要，将边缘叠加到原始图像上
    if args.overlay:
        result = overlay_edges(image, edges)
    else:
        # 如果不叠加，则创建一个三通道的边缘图像以便显示
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 显示结果
    if args.display:
        image_utils.display_image(result, f"边缘检测结果 ({args.method})")
    
    # 保存结果
    if args.output:
        image_utils.save_image(result, args.output)
        print(f"结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 