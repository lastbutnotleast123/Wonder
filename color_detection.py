#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

import image_utils

def detect_color_hsv(image: np.ndarray, 
                    lower_hsv: Tuple[int, int, int], 
                    upper_hsv: Tuple[int, int, int]) -> np.ndarray:
    """
    使用HSV颜色空间检测特定颜色范围
    
    Args:
        image: 输入图像
        lower_hsv: HSV下限 (H, S, V)
        upper_hsv: HSV上限 (H, S, V)
        
    Returns:
        颜色掩码
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    return mask

def apply_color_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    应用颜色掩码到图像
    
    Args:
        image: 输入图像
        mask: 颜色掩码
        
    Returns:
        应用掩码后的图像
    """
    return cv2.bitwise_and(image, image, mask=mask)

def get_color_histogram(image: np.ndarray, mask: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    计算图像的颜色直方图
    
    Args:
        image: 输入图像
        mask: 可选掩码
        
    Returns:
        BGR通道的直方图
    """
    histograms = {}
    color = ('b', 'g', 'r')
    
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
        histograms[col] = hist
    
    return histograms

def plot_color_histogram(histograms: Dict[str, np.ndarray], title: str = "颜色直方图") -> None:
    """
    绘制颜色直方图
    
    Args:
        histograms: BGR通道的直方图
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
    
    for col, hist in histograms.items():
        plt.plot(hist, color=colors[col])
        plt.xlim([0, 256])
    
    plt.title(title)
    plt.xlabel('像素强度')
    plt.ylabel('像素数量')
    plt.show()

def extract_dominant_colors(image: np.ndarray, n_colors: int = 5, 
                           display_result: bool = True) -> List[Tuple[int, int, int]]:
    """
    提取图像中的主要颜色
    
    Args:
        image: 输入图像
        n_colors: 要提取的颜色数量
        display_result: 是否显示结果
        
    Returns:
        主要颜色列表 [(B,G,R), ...]
    """
    # 获取主要颜色
    dominant_colors = image_utils.get_dominant_colors(image, n_colors)
    
    # 如果需要显示结果
    if display_result:
        # 创建一个显示主要颜色的图像
        height = 50
        width = 500
        color_bar = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 计算每种颜色的宽度
        chunk_width = width // n_colors
        
        # 填充颜色条
        for i, color in enumerate(dominant_colors):
            start_x = i * chunk_width
            end_x = (i + 1) * chunk_width
            color_bar[:, start_x:end_x] = color
        
        # 显示颜色条
        plt.figure(figsize=(10, 2))
        plt.imshow(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))
        plt.title("主要颜色")
        plt.axis('off')
        plt.show()
    
    return dominant_colors

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='颜色检测程序')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出图像路径')
    parser.add_argument('--mode', default='dominant', 
                       choices=['dominant', 'specific', 'histogram'], 
                       help='颜色检测模式')
    parser.add_argument('--colors', type=int, default=5, 
                       help='要提取的主要颜色数量')
    parser.add_argument('--lower', nargs=3, type=int, default=[0, 100, 100], 
                       help='HSV下限 (H, S, V)')
    parser.add_argument('--upper', nargs=3, type=int, default=[10, 255, 255], 
                       help='HSV上限 (H, S, V)')
    parser.add_argument('--display', action='store_true', help='是否显示结果')
    args = parser.parse_args()
    
    # 加载图像
    image = image_utils.load_image(args.image)
    
    # 根据选择的模式进行颜色检测
    if args.mode == 'dominant':
        # 提取主要颜色
        dominant_colors = extract_dominant_colors(image, args.colors, args.display)
        print("主要颜色 (BGR):")
        for i, color in enumerate(dominant_colors):
            print(f"颜色 #{i+1}: {color}")
        
        # 保存结果（原始图像）
        if args.output:
            image_utils.save_image(image, args.output)
            print(f"原始图像已保存到: {args.output}")
    
    elif args.mode == 'specific':
        # 检测特定颜色范围
        lower_hsv = tuple(args.lower)
        upper_hsv = tuple(args.upper)
        
        mask = detect_color_hsv(image, lower_hsv, upper_hsv)
        result = apply_color_mask(image, mask)
        
        # 显示结果
        if args.display:
            image_utils.display_image(result, "特定颜色检测结果")
        
        # 保存结果
        if args.output:
            image_utils.save_image(result, args.output)
            print(f"结果已保存到: {args.output}")
    
    elif args.mode == 'histogram':
        # 计算颜色直方图
        histograms = get_color_histogram(image)
        
        # 显示直方图
        if args.display:
            plot_color_histogram(histograms)
        
        # 保存原始图像
        if args.output:
            image_utils.save_image(image, args.output)
            print(f"原始图像已保存到: {args.output}")

if __name__ == "__main__":
    main() 