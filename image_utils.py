#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union

def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        加载的图像数组
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return img

def display_image(image: np.ndarray, title: str = "图像", is_bgr: bool = True) -> None:
    """
    显示图像
    
    Args:
        image: 要显示的图像
        title: 图像标题
        is_bgr: 是否为BGR格式（OpenCV默认格式）
    """
    plt.figure(figsize=(10, 8))
    if is_bgr:
        # 将BGR转换为RGB以便正确显示
        plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        plt_image = image
    plt.imshow(plt_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image: np.ndarray, width: Optional[int] = None, 
                height: Optional[int] = None, 
                scale: Optional[float] = None) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        width: 目标宽度
        height: 目标高度
        scale: 缩放比例
        
    Returns:
        调整大小后的图像
    """
    if scale is not None:
        return cv2.resize(image, None, fx=scale, fy=scale)
    
    if width is not None and height is not None:
        return cv2.resize(image, (width, height))
    
    if width is not None:
        ratio = width / image.shape[1]
        height = int(image.shape[0] * ratio)
        return cv2.resize(image, (width, height))
    
    if height is not None:
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
        return cv2.resize(image, (width, height))
    
    return image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    将图像转换为灰度
    
    Args:
        image: 输入图像
        
    Returns:
        灰度图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                       sigma: float = 0) -> np.ndarray:
    """
    应用高斯模糊
    
    Args:
        image: 输入图像
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
        
    Returns:
        模糊后的图像
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    保存图像到文件
    
    Args:
        image: 要保存的图像
        output_path: 输出文件路径
        
    Returns:
        是否保存成功
    """
    return cv2.imwrite(output_path, image)

def draw_rectangle(image: np.ndarray, start_point: Tuple[int, int], 
                  end_point: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制矩形
    
    Args:
        image: 输入图像
        start_point: 左上角坐标
        end_point: 右下角坐标
        color: 矩形颜色 (B,G,R)
        thickness: 线条粗细
        
    Returns:
        绘制矩形后的图像
    """
    result = image.copy()
    return cv2.rectangle(result, start_point, end_point, color, thickness)

def draw_text(image: np.ndarray, text: str, position: Tuple[int, int], 
             font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0), 
             thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制文本
    
    Args:
        image: 输入图像
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_scale: 字体大小
        color: 文本颜色 (B,G,R)
        thickness: 线条粗细
        
    Returns:
        绘制文本后的图像
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(result, text, position, font, font_scale, color, thickness)

def get_dominant_colors(image: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    获取图像中的主要颜色
    
    Args:
        image: 输入图像
        n_colors: 要提取的颜色数量
        
    Returns:
        主要颜色列表 [(B,G,R), ...]
    """
    # 将图像重塑为像素列表
    pixels = image.reshape(-1, 3)
    
    # 转换为RGB格式以便更直观
    pixels = pixels[:, ::-1]  # BGR到RGB
    
    # 使用K-means聚类找到主要颜色
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    
    # 获取聚类中心（主要颜色）
    colors = kmeans.cluster_centers_
    
    # 转换回BGR格式并转为整数
    colors = colors[:, ::-1].astype(int)
    
    return [tuple(color) for color in colors] 