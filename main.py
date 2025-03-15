#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import time

# 导入自定义模块
import image_utils
import face_detection
import edge_detection
import color_detection
import object_detection
import face_recognition_module
import image_retrieval
import object_tracking
import deep_learning

def create_output_dir(output_dir: str) -> None:
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")

def process_image(image_path: str, output_dir: str, 
                 operations: List[str], display_results: bool = False,
                 **kwargs) -> Dict[str, str]:
    """
    处理图像并应用指定的操作
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        operations: 要应用的操作列表
        display_results: 是否显示结果
        **kwargs: 额外参数
        
    Returns:
        操作结果的文件路径字典
    """
    # 加载图像
    try:
        image = image_utils.load_image(image_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return {}
    
    # 获取图像文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 结果文件路径字典
    result_paths = {}
    
    # 应用每个操作
    for operation in operations:
        if operation == "face_detect":
            # 人脸检测
            faces = face_detection.detect_faces(image)
            result = face_detection.mark_faces(image, faces, True)
            output_path = os.path.join(output_dir, f"{image_name}_face_detect.jpg")
            image_utils.save_image(result, output_path)
            result_paths["face_detect"] = output_path
            
            if display_results:
                image_utils.display_image(result, "人脸检测结果")
            
            print(f"检测到 {len(faces)} 个人脸")
        
        elif operation == "face_recognize":
            # 人脸识别
            recognizer = face_recognition_module.FaceRecognizer()
            face_results = recognizer.recognize_faces(image)
            result = recognizer.mark_faces(image, face_results)
            output_path = os.path.join(output_dir, f"{image_name}_face_recognize.jpg")
            image_utils.save_image(result, output_path)
            result_paths["face_recognize"] = output_path
            
            if display_results:
                image_utils.display_image(result, "人脸识别结果")
            
            print(f"识别到 {len(face_results)} 个人脸")
            for i, ((top, right, bottom, left), name) in enumerate(face_results):
                print(f"人脸 #{i+1}: {name}")
        
        elif operation == "edge":
            # 边缘检测（使用Canny算法）
            edges = edge_detection.detect_edges_canny(image)
            result = edge_detection.overlay_edges(image, edges)
            output_path = os.path.join(output_dir, f"{image_name}_edge.jpg")
            image_utils.save_image(result, output_path)
            result_paths["edge"] = output_path
            
            if display_results:
                image_utils.display_image(result, "边缘检测结果")
        
        elif operation == "color":
            # 颜色分析
            dominant_colors = color_detection.extract_dominant_colors(image, 5, display_results)
            output_path = os.path.join(output_dir, f"{image_name}_color.jpg")
            image_utils.save_image(image, output_path)
            result_paths["color"] = output_path
            
            print("主要颜色 (BGR):")
            for i, color in enumerate(dominant_colors):
                print(f"颜色 #{i+1}: {color}")
        
        elif operation == "object_detect":
            # 物体检测
            try:
                model = object_detection.load_ssd_model()
                detections = object_detection.detect_objects(image, model)
                result = object_detection.draw_detections(image, detections)
                output_path = os.path.join(output_dir, f"{image_name}_object_detect.jpg")
                image_utils.save_image(result, output_path)
                result_paths["object_detect"] = output_path
                
                if display_results:
                    image_utils.display_image(result, "物体检测结果")
                
                print(f"检测到 {len(detections)} 个物体:")
                for i, detection in enumerate(detections):
                    print(f"{i+1}. {detection['class_name']} (置信度: {detection['confidence']:.2f})")
            
            except FileNotFoundError as e:
                print(f"错误: {e}")
        
        elif operation == "image_retrieve":
            # 图像检索
            if 'db_path' in kwargs:
                db = image_retrieval.ImageDatabase(db_path=kwargs['db_path'])
                results = db.search_image(image, kwargs.get('top_k', 5))
                
                if results and display_results:
                    # 获取第一个结果进行可视化
                    best_match_path, similarity, _ = results[0]
                    best_match_image = image_utils.load_image(best_match_path)
                    
                    # 创建特征匹配可视化
                    descriptor = image_retrieval.ImageDescriptor()
                    query_kp, query_desc = descriptor.extract_features(image)
                    match_kp, match_desc = descriptor.extract_features(best_match_image)
                    matches = descriptor.match_features(query_desc, match_desc)
                    
                    match_vis = image_retrieval.visualize_matches(
                        image, best_match_image, query_kp, match_kp, matches)
                    
                    output_path = os.path.join(output_dir, f"{image_name}_retrieve.jpg")
                    image_utils.save_image(match_vis, output_path)
                    result_paths["image_retrieve"] = output_path
                    
                    # 显示结果
                    image_utils.display_image(match_vis, "图像检索结果")
                
                print(f"找到 {len(results)} 个相似图像:")
                for i, (path, sim, _) in enumerate(results):
                    print(f"{i+1}. {path} (相似度: {sim:.2f})")
            else:
                print("错误: 图像检索需要指定数据库路径")
        
        elif operation == "deep_classify":
            # 深度学习图像分类
            if all(k in kwargs for k in ['model_path', 'classes_file']):
                try:
                    classifier = deep_learning.ImageClassifier(
                        kwargs['model_path'], 
                        kwargs.get('config_path'), 
                        kwargs.get('framework', 'tensorflow'),
                        kwargs['classes_file']
                    )
                    
                    # 分类图像
                    results = classifier.classify(image, kwargs.get('top_k', 5))
                    
                    if results:
                        # 在图像上显示结果
                        result_image = image.copy()
                        for i, (class_id, class_name, probability) in enumerate(results):
                            text = f"{class_name}: {probability:.2f}"
                            cv2.putText(result_image, text, (10, 30 + i * 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # 保存和显示结果
                        output_path = os.path.join(output_dir, f"{image_name}_classify.jpg")
                        image_utils.save_image(result_image, output_path)
                        result_paths["deep_classify"] = output_path
                        
                        if display_results:
                            image_utils.display_image(result_image, "分类结果")
                        
                        # 打印分类结果
                        print("分类结果:")
                        for i, (class_id, class_name, probability) in enumerate(results):
                            print(f"#{i+1}: {class_name} (概率: {probability:.4f})")
                    else:
                        print("未能获得分类结果")
                        
                except Exception as e:
                    print(f"分类过程中出错: {e}")
            else:
                print("错误: 深度学习分类需要指定模型路径和类别文件")
        
        elif operation == "deep_detect":
            # 深度学习物体检测
            if all(k in kwargs for k in ['model_path', 'config_path']):
                try:
                    detector = deep_learning.ObjectDetector(
                        kwargs['model_path'], 
                        kwargs['config_path'],
                        kwargs.get('framework', 'tensorflow'),
                        kwargs.get('classes_file')
                    )
                    
                    # 检测物体
                    detections = detector.detect(image, kwargs.get('confidence', 0.5))
                    
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
                        
                        # 保存和显示结果
                        output_path = os.path.join(output_dir, f"{image_name}_deep_detect.jpg")
                        image_utils.save_image(result_image, output_path)
                        result_paths["deep_detect"] = output_path
                        
                        if display_results:
                            image_utils.display_image(result_image, "深度学习检测结果")
                        
                        # 打印检测结果
                        print(f"深度学习检测到 {len(detections)} 个物体:")
                        for i, detection in enumerate(detections):
                            print(f"#{i+1}: {detection['class_name']} (置信度: {detection['confidence']:.4f})")
                    else:
                        print("未检测到物体")
                        
                except Exception as e:
                    print(f"检测过程中出错: {e}")
            else:
                print("错误: 深度学习检测需要指定模型路径和配置文件")
        
        elif operation == "segment":
            # 图像分割
            if all(k in kwargs for k in ['model_path', 'config_path']):
                try:
                    segmenter = deep_learning.ImageSegmenter(
                        kwargs['model_path'], 
                        kwargs['config_path'],
                        kwargs.get('framework', 'tensorflow'),
                        kwargs.get('classes_file')
                    )
                    
                    # 分割图像
                    mask, vis_image = segmenter.segment(image)
                    
                    # 保存和显示结果
                    output_path = os.path.join(output_dir, f"{image_name}_segment.jpg")
                    image_utils.save_image(vis_image, output_path)
                    result_paths["segment"] = output_path
                    
                    if display_results:
                        image_utils.display_image(vis_image, "分割结果")
                    
                    # 打印统计信息
                    print("分割结果:")
                    classes_count = {}
                    for i in range(np.max(mask) + 1):
                        count = np.sum(mask == i)
                        if count > 0:
                            class_name = segmenter.classes[i] if i < len(segmenter.classes) else f"类别{i}"
                            percentage = count / (mask.shape[0] * mask.shape[1]) * 100
                            classes_count[class_name] = percentage
                            print(f"类别 {i} ({class_name}): {percentage:.2f}%")
                        
                except Exception as e:
                    print(f"分割过程中出错: {e}")
            else:
                print("错误: 图像分割需要指定模型路径和配置文件")
        
        elif operation == "all":
            # 应用所有基本操作
            operations.extend(["face_detect", "edge", "color", "object_detect"])
            # 移除重复项
            operations = list(set(operations))
            # 移除 "all"
            operations.remove("all")
            # 递归调用
            return process_image(image_path, output_dir, operations, display_results)
    
    return result_paths

def process_video(video_path: str, output_dir: str, 
                operation: str, display_results: bool = False,
                **kwargs) -> Optional[str]:
    """
    处理视频并应用指定的操作
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        operation: 要应用的操作
        display_results: 是否显示结果
        **kwargs: 额外参数
        
    Returns:
        输出视频路径
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 输出视频路径
    output_path = os.path.join(output_dir, f"{video_name}_{operation}.avi")
    
    if operation == "object_track":
        # 目标跟踪
        tracker_type = kwargs.get('tracker_type', 'csrt')
        object_tracking.process_video(video_path, tracker_type, output_path, display_results)
        return output_path
    
    elif operation == "face_track":
        # 人脸跟踪（TODO: 实现人脸跟踪功能）
        print("人脸跟踪功能尚未实现")
        return None
    
    else:
        print(f"不支持的视频处理操作: {operation}")
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基于OpenCV的图像识别程序')
    parser.add_argument('--input', required=True, help='输入图像或视频路径')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--mode', choices=['image', 'video'], default='image',
                       help='处理模式：图像(image)或视频(video)')
    parser.add_argument('--operations', nargs='+', 
                       choices=['all', 'face_detect', 'face_recognize', 'edge', 'color', 
                               'object_detect', 'image_retrieve', 'deep_classify', 
                               'deep_detect', 'segment', 'object_track', 'face_track'],
                       default=['all'], 
                       help='要应用的操作')
    parser.add_argument('--display', action='store_true', help='是否显示结果')
    
    # 高级参数
    parser.add_argument('--db_path', help='图像检索数据库路径')
    parser.add_argument('--model_path', help='深度学习模型路径')
    parser.add_argument('--config_path', help='深度学习模型配置路径')
    parser.add_argument('--classes_file', help='类别文件路径')
    parser.add_argument('--framework', choices=['tensorflow', 'caffe', 'darknet', 'torch'],
                       default='tensorflow', help='深度学习框架')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--top_k', type=int, default=5, help='返回的结果数量')
    parser.add_argument('--tracker_type', choices=list(object_tracking.ObjectTracker.TRACKER_TYPES.keys()),
                       default='csrt', help='目标跟踪器类型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    create_output_dir(args.output)
    
    # 收集额外参数
    extra_params = {
        'db_path': args.db_path,
        'model_path': args.model_path,
        'config_path': args.config_path,
        'classes_file': args.classes_file,
        'framework': args.framework,
        'confidence': args.confidence,
        'top_k': args.top_k,
        'tracker_type': args.tracker_type
    }
    
    # 过滤掉None值
    extra_params = {k: v for k, v in extra_params.items() if v is not None}
    
    # 根据模式处理输入
    if args.mode == 'image':
        # 处理图像
        result_paths = process_image(args.input, args.output, args.operations, args.display, **extra_params)
        
        if result_paths:
            print("\n处理完成！结果已保存到以下文件:")
            for operation, path in result_paths.items():
                print(f"- {operation}: {path}")
        else:
            print("处理过程中出错，没有生成结果")
    
    elif args.mode == 'video':
        # 处理视频
        if len(args.operations) != 1:
            print("视频模式下只能指定一个操作")
            return
        
        operation = args.operations[0]
        output_path = process_video(args.input, args.output, operation, args.display, **extra_params)
        
        if output_path:
            print(f"\n处理完成！结果已保存到: {output_path}")
        else:
            print("处理过程中出错，没有生成结果")

if __name__ == "__main__":
    main() 