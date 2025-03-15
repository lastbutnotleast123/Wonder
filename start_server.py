#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启动服务器的简化脚本
"""

import os
import uuid
import base64
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, send_from_directory, Blueprint, jsonify, request, redirect
from flask_cors import CORS
import api
from api import api_bp
import image_retrieval

# 定义所需的目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_output')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_interface')

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 创建Flask应用
app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)  # 启用跨域请求

# 配置上传文件夹和最大文件大小
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 创建必要的目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """保存上传的文件并返回文件路径"""
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        return filepath
    return None

def save_base64_image(base64_data):
    """保存Base64编码的图像并返回文件路径"""
    try:
        # 移除可能的前缀
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        # 解码Base64数据
        image_data = base64.b64decode(base64_data)
        
        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # 保存图像数据
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filepath
    except Exception as e:
        print(f"Base64图像保存错误: {e}")
        return None

def get_output_path(input_path, operation):
    """生成输出文件路径"""
    base_name = os.path.basename(input_path)
    output_name = f"{operation}_{base_name}"
    return os.path.join(OUTPUT_FOLDER, output_name)

def image_to_base64(image_path):
    """将图像转换为Base64编码"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Base64编码错误: {e}")
        return None

def process_basic_image(input_path, output_path):
    """基本图像处理：调整大小、亮度等"""
    try:
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图像"
            
        # 调整图像大小
        height, width = img.shape[:2]
        max_size = 800
        if height > max_size or width > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            img = cv2.resize(img, (new_width, new_height))
            
        # 轻微提高对比度
        alpha = 1.1  # 对比度控制
        beta = 10    # 亮度控制
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 保存处理后的图像
        cv2.imwrite(output_path, img)
        return True, "基本处理完成"
    except Exception as e:
        print(f"处理图像错误: {e}")
        return False, str(e)

def detect_faces(input_path, output_path):
    """人脸检测"""
    try:
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图像"
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 加载人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 在图像上标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        # 保存结果
        cv2.imwrite(output_path, img)
        
        return True, f"检测到 {len(faces)} 个人脸"
    except Exception as e:
        print(f"人脸检测错误: {e}")
        return False, str(e)

def detect_edges(input_path, output_path, method='canny'):
    """边缘检测"""
    try:
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图像"
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 根据方法选择相应的边缘检测方式
        if method == 'canny':
            edges = cv2.Canny(gray, 100, 200)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(sobelx, sobely)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            edges = cv2.Canny(gray, 100, 200)  # 默认使用Canny
        
        # 将边缘叠加到原图
        result = img.copy()
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 0.7, edges_colored, 0.3, 0)
        
        # 保存结果
        cv2.imwrite(output_path, result)
        
        return True, f"使用{method}方法完成边缘检测"
    except Exception as e:
        print(f"边缘检测错误: {e}")
        return False, str(e)

def analyze_colors(input_path, output_path):
    """颜色分析"""
    try:
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图像"
            
        # 获取主要颜色
        pixels = img.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5  # 主要颜色数量
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 转换回整数
        centers = np.uint8(centers)
        
        # 创建一个显示主要颜色的图像
        height = 50
        width = img.shape[1]
        color_image = np.zeros((height * k, width, 3), dtype=np.uint8)
        
        # 统计每种颜色的像素数量
        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(counts)[::-1]  # 从大到小排序
        
        # 在图像上显示每种主要颜色
        y_pos = 0
        for idx in sorted_indices[:k]:
            color = centers[idx]
            percentage = counts[idx] / len(labels) * 100
            color_image[y_pos:y_pos+height, :] = color
            y_pos += height
            
        # 将结果图像保存
        cv2.imwrite(output_path, color_image)
        
        return True, "颜色分析完成"
    except Exception as e:
        print(f"颜色分析错误: {e}")
        return False, str(e)

def detect_objects(input_path, output_path):
    """简单物体检测"""
    try:
        # 由于真正的物体检测需要预训练模型，这里我们使用轮廓检测作为简化示例
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图像"
            
        # 转换为灰度图并模糊处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        min_area = 500
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # 在原图上绘制轮廓
        result = img.copy()
        cv2.drawContours(result, significant_contours, -1, (0, 255, 0), 2)
        
        # 保存结果
        cv2.imwrite(output_path, result)
        
        return True, f"检测到 {len(significant_contours)} 个物体"
    except Exception as e:
        print(f"物体检测错误: {e}")
        return False, str(e)

# 注册API蓝图
app.register_blueprint(api_bp, url_prefix='/api')

# 初始化图像检索系统
image_retrieval.initialize()

@app.route('/')
def home():
    return redirect('/index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_FOLDER, path)

if __name__ == '__main__':
    print("启动简化服务器...")
    print("请在浏览器中访问: http://localhost:5001")
    print("API状态检查: http://localhost:5001/api/status")
    app.run(debug=True, host='0.0.0.0', port=5001) 