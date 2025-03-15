#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import face_recognition
import os
import pickle
import argparse
from typing import List, Tuple, Dict, Optional, Union
import time

import image_utils

class FaceRecognizer:
    """人脸识别器类"""
    
    def __init__(self, model_path: Optional[str] = None, 
                encoding_path: Optional[str] = None):
        """
        初始化人脸识别器
        
        Args:
            model_path: 人脸识别模型路径（可选）
            encoding_path: 人脸编码数据路径（可选）
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.encoding_path = encoding_path or "models/face_encodings.pkl"
        
        # 如果存在编码文件，则加载
        if os.path.exists(self.encoding_path):
            self.load_encodings()
    
    def load_encodings(self) -> None:
        """加载已保存的人脸编码"""
        try:
            with open(self.encoding_path, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print(f"已加载 {len(self.known_face_names)} 个人脸编码")
        except Exception as e:
            print(f"加载人脸编码时出错: {e}")
    
    def save_encodings(self) -> None:
        """保存人脸编码到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.encoding_path), exist_ok=True)
        
        # 保存编码
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        with open(self.encoding_path, "wb") as f:
            pickle.dump(data, f)
        print(f"已保存 {len(self.known_face_names)} 个人脸编码到 {self.encoding_path}")
    
    def add_face(self, image: np.ndarray, name: str) -> bool:
        """
        添加新人脸
        
        Args:
            image: 包含人脸的图像
            name: 人脸对应的名称
            
        Returns:
            是否成功添加
        """
        # 检测人脸位置
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            print("未检测到人脸")
            return False
        
        # 使用第一个检测到的人脸
        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        
        # 检查是否已存在此人脸
        if len(self.known_face_encodings) > 0:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                if self.known_face_names[match_index] == name:
                    print(f"此人脸 ({name}) 已存在")
                    return False
        
        # 添加新人脸
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        self.save_encodings()
        print(f"已添加新人脸: {name}")
        return True
    
    def recognize_faces(self, image: np.ndarray, 
                       tolerance: float = 0.6) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        识别图像中的人脸
        
        Args:
            image: 输入图像
            tolerance: 人脸匹配容差，值越小越严格
            
        Returns:
            人脸位置和名称的列表 [((top, right, bottom, left), name), ...]
        """
        # 如果没有已知人脸，返回空列表
        if not self.known_face_encodings:
            print("没有已知人脸可供匹配")
            return []
        
        # 将图像转换为RGB（face_recognition需要RGB格式）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测所有人脸位置
        face_locations = face_recognition.face_locations(rgb_image)
        
        # 如果未检测到人脸，返回空列表
        if not face_locations:
            return []
        
        # 提取所有人脸的编码
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # 匹配结果列表
        results = []
        
        # 遍历每个检测到的人脸编码
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 与已知人脸比较
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance)
            
            # 默认为"未知"
            name = "未知"
            
            # 查找最佳匹配
            if True in matches:
                # 找出所有匹配的索引
                matched_indices = [i for i, match in enumerate(matches) if match]
                
                if matched_indices:
                    # 计算与每个匹配的人脸的距离
                    face_distances = face_recognition.face_distance(
                        [self.known_face_encodings[i] for i in matched_indices], 
                        face_encoding
                    )
                    # 选择距离最小的人脸
                    best_match_index = matched_indices[np.argmin(face_distances)]
                    name = self.known_face_names[best_match_index]
            
            # 添加结果
            results.append(((top, right, bottom, left), name))
        
        return results
    
    def mark_faces(self, image: np.ndarray, 
                  face_results: List[Tuple[Tuple[int, int, int, int], str]]) -> np.ndarray:
        """
        在图像上标记识别结果
        
        Args:
            image: 输入图像
            face_results: 识别结果列表 [((top, right, bottom, left), name), ...]
            
        Returns:
            标记后的图像
        """
        result = image.copy()
        
        for (top, right, bottom, left), name in face_results:
            # 绘制人脸矩形
            cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # 绘制名称标签背景
            cv2.rectangle(result, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            # 绘制名称
            cv2.putText(result, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人脸识别程序")
    parser.add_argument("--mode", choices=["recognize", "add"], default="recognize",
                       help="操作模式：识别(recognize)或添加(add)人脸")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--name", help="添加人脸时的名称")
    parser.add_argument("--output", help="输出图像路径")
    parser.add_argument("--display", action="store_true", help="是否显示结果")
    parser.add_argument("--tolerance", type=float, default=0.6, help="人脸匹配容差，值越小越严格")
    args = parser.parse_args()
    
    # 加载图像
    image = image_utils.load_image(args.image)
    
    # 创建人脸识别器
    recognizer = FaceRecognizer()
    
    if args.mode == "add":
        # 添加人脸模式
        if not args.name:
            print("错误：添加人脸时必须提供名称")
            return
        
        result = recognizer.add_face(image, args.name)
        
        if result:
            print(f"已成功添加 {args.name} 的人脸")
        else:
            print("添加人脸失败")
    
    else:  # recognize 模式
        # 开始计时
        start_time = time.time()
        
        # 识别人脸
        face_results = recognizer.recognize_faces(image, args.tolerance)
        
        # 计算用时
        elapsed_time = time.time() - start_time
        
        # 标记识别结果
        result_image = recognizer.mark_faces(image, face_results)
        
        # 显示结果
        if args.display:
            image_utils.display_image(result_image, "人脸识别结果")
        
        # 保存结果
        if args.output:
            image_utils.save_image(result_image, args.output)
            print(f"结果已保存到: {args.output}")
        
        # 打印识别结果
        print(f"识别完成，共检测到 {len(face_results)} 个人脸，用时 {elapsed_time:.2f} 秒")
        for i, ((top, right, bottom, left), name) in enumerate(face_results):
            print(f"人脸 #{i+1}: {name} 位置: 左={left}, 上={top}, 右={right}, 下={bottom}")

if __name__ == "__main__":
    main() 