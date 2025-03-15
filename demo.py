#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示脚本：展示OpenCV图像处理与计算机视觉项目的主要功能
"""

import os
import argparse
import subprocess
import time
import shutil

def create_output_dir(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_command(command, description):
    """运行命令并显示描述"""
    print("\n" + "="*80)
    print(f"演示: {description}")
    print("命令: " + command)
    print("="*80)
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ {description}执行成功!\n")
        time.sleep(1)  # 暂停一下，让用户有时间查看结果
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description}执行失败: {e}\n")

def demo_basic_image_processing(image_path, output_dir):
    """演示基本图像处理功能"""
    command = f"python main.py --mode image --input {image_path} --operations all --output {output_dir}/basic --display"
    run_command(command, "基本图像处理")

def demo_face_detection(image_path, output_dir):
    """演示人脸检测功能"""
    command = f"python face_detection.py --image {image_path} --output {output_dir}/face_detection.jpg --eyes --display"
    run_command(command, "人脸检测")

def demo_edge_detection(image_path, output_dir):
    """演示边缘检测功能"""
    # 尝试不同的边缘检测算法
    for method in ["canny", "sobel", "laplacian"]:
        command = f"python edge_detection.py --image {image_path} --output {output_dir}/edge_{method}.jpg --method {method} --display"
        run_command(command, f"边缘检测 ({method})")

def demo_color_detection(image_path, output_dir):
    """演示颜色检测功能"""
    # 主要颜色提取
    command = f"python color_detection.py --image {image_path} --output {output_dir}/color_dominant.jpg --mode dominant --display"
    run_command(command, "主要颜色提取")
    
    # 颜色直方图
    command = f"python color_detection.py --image {image_path} --output {output_dir}/color_histogram.jpg --mode histogram --display"
    run_command(command, "颜色直方图")

def demo_object_detection(image_path, output_dir):
    """演示物体检测功能"""
    command = f"python object_detection.py --image {image_path} --output {output_dir}/object_detection.jpg --display"
    run_command(command, "物体检测")

def demo_face_recognition(face_image, output_dir):
    """演示人脸识别功能"""
    # 添加人脸到数据库
    command = f"python face_recognition_module.py --mode add --image {face_image} --name \"示例人脸\""
    run_command(command, "添加人脸到数据库")
    
    # 识别人脸
    command = f"python face_recognition_module.py --mode recognize --image {face_image} --output {output_dir}/face_recognition.jpg --display"
    run_command(command, "人脸识别")

def demo_image_retrieval(images_dir, query_image, output_dir):
    """演示图像检索功能"""
    # 批量添加图像到数据库
    command = f"python image_retrieval.py --mode batch_add --pattern \"{images_dir}/*.jpg\""
    run_command(command, "构建图像数据库")
    
    # 搜索相似图像
    command = f"python image_retrieval.py --mode search --image {query_image} --output {output_dir}/image_retrieval.jpg --display"
    run_command(command, "图像检索")

def demo_object_tracking(video_path, output_dir):
    """演示物体跟踪功能"""
    command = f"python object_tracking.py --video {video_path} --output {output_dir}/tracking.avi --tracker csrt --display"
    run_command(command, "物体跟踪")

def demo_deep_learning(image_path, model_path, config_path, classes_file, output_dir):
    """演示深度学习功能"""
    # 图像分类
    if os.path.exists(model_path) and os.path.exists(classes_file):
        command = f"python deep_learning.py --mode classify --image {image_path} --model {model_path} --classes {classes_file} --output {output_dir}/deep_classify.jpg --display"
        run_command(command, "深度学习图像分类")
    
    # 物体检测
    if os.path.exists(model_path) and os.path.exists(config_path):
        command = f"python deep_learning.py --mode detect --image {image_path} --model {model_path} --config {config_path} --output {output_dir}/deep_detect.jpg --display"
        run_command(command, "深度学习物体检测")

def main():
    parser = argparse.ArgumentParser(description="OpenCV图像处理与计算机视觉项目演示")
    parser.add_argument("--image", default="images/sample.jpg", help="测试图像路径")
    parser.add_argument("--face_image", default="images/face.jpg", help="测试人脸图像路径")
    parser.add_argument("--video", default="videos/sample.mp4", help="测试视频路径")
    parser.add_argument("--images_dir", default="images", help="测试图像目录")
    parser.add_argument("--model", default="models/mobilenet_v2.pb", help="深度学习模型路径")
    parser.add_argument("--config", default="models/mobilenet_v2.pbtxt", help="深度学习配置文件")
    parser.add_argument("--classes", default="models/classes.txt", help="类别文件")
    parser.add_argument("--output", default="demo_output", help="演示输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_output_dir(args.output)
    
    print("\n" + "*"*80)
    print("*" + " "*30 + "OpenCV项目功能演示" + " "*30 + "*")
    print("*" + " "*78 + "*")
    print("* 本演示将展示项目的主要功能，包括：" + " "*39 + "*")
    print("* - 基本图像处理" + " "*61 + "*")
    print("* - 人脸检测与识别" + " "*59 + "*")
    print("* - 边缘检测" + " "*65 + "*")
    print("* - 颜色分析" + " "*65 + "*")
    print("* - 物体检测" + " "*65 + "*")
    print("* - 图像检索" + " "*65 + "*")
    print("* - 物体跟踪" + " "*65 + "*")
    print("* - 深度学习应用" + " "*61 + "*")
    print("*" + " "*78 + "*")
    print("* 注意：某些功能可能需要额外的数据文件或模型" + " "*36 + "*")
    print("*" + " "*78 + "*")
    print("*"*80)
    
    # 检查示例文件是否存在
    if not os.path.exists(args.image):
        print(f"警告: 示例图像 {args.image} 不存在")
        return
    
    # 演示基本图像处理
    demo_basic_image_processing(args.image, output_dir)
    
    # 演示人脸检测
    if os.path.exists(args.face_image):
        demo_face_detection(args.face_image, output_dir)
    else:
        print(f"警告: 人脸图像 {args.face_image} 不存在，跳过人脸检测演示")
    
    # 演示边缘检测
    demo_edge_detection(args.image, output_dir)
    
    # 演示颜色检测
    demo_color_detection(args.image, output_dir)
    
    # 演示物体检测
    demo_object_detection(args.image, output_dir)
    
    # 演示人脸识别
    if os.path.exists(args.face_image):
        demo_face_recognition(args.face_image, output_dir)
    else:
        print(f"警告: 人脸图像 {args.face_image} 不存在，跳过人脸识别演示")
    
    # 演示图像检索
    if os.path.exists(args.images_dir) and len(os.listdir(args.images_dir)) > 1:
        demo_image_retrieval(args.images_dir, args.image, output_dir)
    else:
        print(f"警告: 图像目录 {args.images_dir} 不存在或图像不足，跳过图像检索演示")
    
    # 演示物体跟踪
    if os.path.exists(args.video):
        demo_object_tracking(args.video, output_dir)
    else:
        print(f"警告: 视频文件 {args.video} 不存在，跳过物体跟踪演示")
    
    # 演示深度学习
    demo_deep_learning(args.image, args.model, args.config, args.classes, output_dir)
    
    print("\n" + "="*80)
    print(f"演示完成! 所有结果已保存到 {output_dir} 目录")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 