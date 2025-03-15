#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RESTful API接口：提供HTTP接口调用图像处理功能
使用Flask框架实现
"""

import os
import uuid
import json
import base64
import tempfile
import argparse
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback

from flask import Flask, request, jsonify, send_file, Blueprint, current_app
from flask_cors import CORS

# 导入项目模块
import image_utils
from face_detection import detect_faces
from edge_detection import detect_edges_canny, detect_edges_sobel, detect_edges_laplacian, overlay_edges
from color_detection import detect_color_hsv, extract_dominant_colors, plot_color_histogram, get_color_histogram
from object_detection import detect_objects
# 暂时注释掉 face_recognition 相关导入
# from face_recognition_module import FaceRecognizer
from image_retrieval import ImageDatabase, ImageDescriptor, retrieval_system
from deep_learning import ImageClassifier, ObjectDetector, ImageSegmenter

# 创建Blueprint而不是直接使用app
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 配置上传文件存储
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
OUTPUT_FOLDER = 'api_output'

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 创建Flask应用并配置
def create_app():
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求
    app.register_blueprint(api_bp)
    
    # 配置上传文件存储
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
    
    return app

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

def get_output_path(filename, operation):
    """生成输出文件路径"""
    base_name = os.path.basename(filename)
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

def generate_color_histogram(input_path, output_path):
    """生成颜色直方图并保存为图像"""
    # 加载图像
    image = cv2.imread(input_path)
    
    # 获取直方图数据
    histograms = get_color_histogram(image)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
    
    for col, hist in histograms.items():
        plt.plot(hist, color=colors[col])
    
    plt.title("颜色直方图")
    plt.xlabel("像素值")
    plt.ylabel("频率")
    plt.xlim([0, 256])
    plt.grid()
    
    # 保存图形
    plt.savefig(output_path)
    plt.close()

@api_bp.route('/status', methods=['GET'])
def status():
    """API状态检查"""
    return jsonify({
        'status': 'active',
        'message': 'OpenCV图像处理API服务正常'
    })

@api_bp.route('/process', methods=['POST'])
def process_image():
    """处理上传的图像"""
    # 检查是否有文件或Base64数据
    if 'image' in request.files:
        file = request.files['image']
        input_path = save_uploaded_file(file)
    elif request.json and 'image_base64' in request.json:
        input_path = save_base64_image(request.json['image_base64'])
    else:
        return jsonify({'error': '未找到图像文件或Base64数据'}), 400
    
    if not input_path:
        return jsonify({'error': '无效的图像格式'}), 400
    
    # 获取请求的操作类型
    operations = request.form.get('operations', '') if request.form else ''
    if request.json:
        operations = request.json.get('operations', '')
    
    if not operations:
        operations = 'basic'  # 默认操作

    # 解析操作列表    
    op_list = operations.split(',')
    results = {}
    
    try:
        # 处理请求的每个操作
        for op in op_list:
            op = op.strip()
            output_path = get_output_path(input_path, op)
            
            if op == 'basic' or op == 'all':
                # 基本图像处理
                img = image_utils.load_image(input_path)
                resized = image_utils.resize_image(img, width=600)
                image_utils.save_image(resized, output_path)
                results[op] = {
                    'success': True,
                    'result_image': image_to_base64(output_path)
                }
            
            elif op == 'face_detect':
                # 人脸检测
                detect_faces(input_path, output_path, detect_eyes=True, display=False)
                results[op] = {
                    'success': True,
                    'result_image': image_to_base64(output_path)
                }
            
            elif op == 'face_recognize':
                # 人脸识别
                # 创建人脸识别器实例
                # recognizer = FaceRecognizer()
                
                # 加载图像
                image = cv2.imread(input_path)
                
                # 识别人脸
                # face_results = recognizer.recognize_faces(image)
                
                # 标记人脸
                # result_image = recognizer.mark_faces(image, face_results)
                
                # 保存结果
                # cv2.imwrite(output_path, result_image)
                
                # 准备返回的人脸信息
                # faces_info = []
                # for (top, right, bottom, left), name in face_results:
                #     faces_info.append({
                #         'name': name,
                #         'position': {'top': top, 'right': right, 'bottom': bottom, 'left': left}
                #     })
                
                results[op] = {
                    'success': True,
                    'faces': [],
                    'result_image': image_to_base64(output_path)
                }
            
            elif op == 'edge':
                # 边缘检测
                method = request.form.get('edge_method', 'canny') if request.form else 'canny'
                if request.json:
                    method = request.json.get('edge_method', 'canny')
                
                # 加载图像
                image = cv2.imread(input_path)
                
                # 根据方法选择相应的边缘检测函数
                if method == 'canny':
                    edges = detect_edges_canny(image)
                elif method == 'sobel':
                    edges = detect_edges_sobel(image)
                elif method == 'laplacian':
                    edges = detect_edges_laplacian(image)
                else:
                    edges = detect_edges_canny(image)  # 默认使用Canny
                
                # 叠加边缘到原图
                result = overlay_edges(image, edges)
                
                # 保存结果
                cv2.imwrite(output_path, result)
                
                results[op] = {
                    'success': True,
                    'method': method,
                    'result_image': image_to_base64(output_path)
                }
            
            elif op == 'color':
                # 颜色分析
                mode = request.form.get('color_mode', 'dominant') if request.form else 'dominant'
                if request.json:
                    mode = request.json.get('color_mode', 'dominant')
                
                if mode == 'dominant':
                    num_colors = int(request.form.get('num_colors', '5')) if request.form else 5
                    if request.json:
                        num_colors = int(request.json.get('num_colors', '5'))
                    
                    colors = extract_dominant_colors(input_path, num_colors)
                    results[op] = {
                        'success': True,
                        'mode': mode,
                        'colors': [{'rgb': c, 'hex': f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'} for c in colors]
                    }
                
                elif mode == 'histogram':
                    generate_color_histogram(input_path, output_path)
                    results[op] = {
                        'success': True,
                        'mode': mode,
                        'result_image': image_to_base64(output_path)
                    }
            
            elif op == 'object_detect':
                # 物体检测
                confidence = float(request.form.get('confidence', '0.5')) if request.form else 0.5
                if request.json:
                    confidence = float(request.json.get('confidence', '0.5'))
                
                detections = detect_objects(input_path, output_path, confidence=confidence, display=False)
                results[op] = {
                    'success': True,
                    'detections': detections,
                    'result_image': image_to_base64(output_path)
                }
            
            elif op == 'image_retrieve':
                # 图像检索
                db_path = request.form.get('db_path') if request.form else None
                if request.json:
                    db_path = request.json.get('db_path')
                
                top_k = int(request.form.get('top_k', '5')) if request.form else 5
                if request.json:
                    top_k = int(request.json.get('top_k', '5'))
                
                if db_path and os.path.exists(db_path):
                    # 创建图像数据库实例
                    descriptor_method = request.form.get('descriptor', 'sift') if request.form else 'sift'
                    if request.json:
                        descriptor_method = request.json.get('descriptor', 'sift')
                    
                    image_db = ImageDatabase(descriptor_method, db_path)
                    
                    # 加载查询图像
                    query_image = cv2.imread(input_path)
                    
                    # 搜索相似图像
                    similar_images = image_db.search_image(query_image, top_k)
                    
                    # 准备返回的匹配信息
                    matches = []
                    for image_path, similarity, metadata in similar_images:
                        matches.append({
                            'image_path': image_path,
                            'similarity': similarity,
                            'metadata': metadata
                        })
                    
                    # 如果需要可视化结果
                    if len(similar_images) > 0:
                        # 获取第一个匹配结果的图像
                        best_match_path = similar_images[0][0]
                        best_match_img = cv2.imread(best_match_path)
                        
                        # 创建描述符提取器
                        descriptor = ImageDescriptor(descriptor_method)
                        
                        # 提取特征点和描述符
                        query_kp, query_desc = descriptor.extract_features(query_image)
                        match_kp, match_desc = descriptor.extract_features(best_match_img)
                        
                        # 匹配特征
                        good_matches = descriptor.match_features(query_desc, match_desc)
                        
                        # 绘制匹配
                        from image_retrieval import visualize_matches
                        result_image = visualize_matches(query_image, best_match_img, 
                                                       query_kp, match_kp, good_matches)
                        
                        # 保存结果
                        cv2.imwrite(output_path, result_image)
                    
                    results[op] = {
                        'success': True,
                        'matches': matches,
                        'result_image': image_to_base64(output_path) if os.path.exists(output_path) else None
                    }
                else:
                    results[op] = {
                        'success': False,
                        'error': '未找到图像数据库或数据库路径未提供'
                    }
            
            elif op in ['deep_classify', 'deep_detect', 'segment']:
                # 深度学习操作
                model_path = request.form.get('model_path') if request.form else None
                config_path = request.form.get('config_path') if request.form else None
                classes_file = request.form.get('classes_file') if request.form else None
                framework = request.form.get('framework', 'tensorflow') if request.form else 'tensorflow'
                
                if request.json:
                    model_path = request.json.get('model_path')
                    config_path = request.json.get('config_path')
                    classes_file = request.json.get('classes_file')
                    framework = request.json.get('framework', 'tensorflow')
                
                if not model_path or not os.path.exists(model_path):
                    results[op] = {
                        'success': False,
                        'error': '未找到模型文件或模型路径未提供'
                    }
                    continue
                
                if op == 'deep_classify' and classes_file and os.path.exists(classes_file):
                    # 创建分类器实例
                    classifier = ImageClassifier(model_path, config_path, framework, classes_file)
                    
                    # 加载图像
                    image = cv2.imread(input_path)
                    
                    # 进行分类
                    predictions = classifier.classify(image, top_k=5)
                    
                    # 将结果保存到输出路径（如有必要）
                    if output_path:
                        # 复制原始图像
                        result_image = image.copy()
                        
                        # 添加分类结果文本
                        y_offset = 30
                        for i, (class_id, class_name, prob) in enumerate(predictions[:3]):
                            text = f"{class_name}: {prob:.2f}"
                            cv2.putText(result_image, text, (10, y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_offset += 30
                        
                        # 保存结果
                        cv2.imwrite(output_path, result_image)
                    
                    # 准备返回结果
                    results[op] = {
                        'success': True,
                        'predictions': [{'class_id': int(c_id), 'class_name': name, 'probability': float(prob)} 
                                     for c_id, name, prob in predictions],
                        'result_image': image_to_base64(output_path) if os.path.exists(output_path) else None
                    }
                
                elif op == 'deep_detect' and config_path and os.path.exists(config_path):
                    confidence = float(request.form.get('confidence', '0.5')) if request.form else 0.5
                    if request.json:
                        confidence = float(request.json.get('confidence', '0.5'))
                    
                    # 创建检测器实例
                    detector = ObjectDetector(model_path, config_path, framework, classes_file)
                    
                    # 加载图像
                    image = cv2.imread(input_path)
                    
                    # 进行检测
                    detections = detector.detect(image, confidence_threshold=confidence)
                    
                    # 可视化结果并保存
                    if output_path:
                        # 复制原始图像
                        result_image = image.copy()
                        
                        # 在图像上绘制检测结果
                        for obj in detections:
                            x1, y1, x2, y2 = obj['box']
                            label = obj['label']
                            confidence = obj['confidence']
                            
                            # 绘制边界框
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 绘制标签
                            text = f"{label}: {confidence:.2f}"
                            cv2.putText(result_image, text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 保存结果
                        cv2.imwrite(output_path, result_image)
                    
                    results[op] = {
                        'success': True,
                        'detections': detections,
                        'result_image': image_to_base64(output_path) if os.path.exists(output_path) else None
                    }
                
                elif op == 'segment' and config_path and os.path.exists(config_path):
                    # 创建分割器实例
                    segmenter = ImageSegmenter(model_path, config_path, framework, classes_file)
                    
                    # 加载图像
                    image = cv2.imread(input_path)
                    
                    # 进行分割
                    mask, class_map = segmenter.segment(image)
                    
                    # 可视化结果并保存
                    if output_path:
                        # 创建彩色掩码
                        colored_mask = np.zeros_like(image)
                        
                        # 为每个类别分配一个颜色（简单实现）
                        unique_classes = np.unique(class_map)
                        for cls in unique_classes:
                            if cls == 0:  # 通常背景类
                                continue
                                
                            # 为每个类生成一个随机颜色
                            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                            colored_mask[class_map == cls] = color
                        
                        # 将分割结果叠加到原图
                        alpha = 0.5
                        beta = 1.0 - alpha
                        result_image = cv2.addWeighted(image, alpha, colored_mask, beta, 0)
                        
                        # 保存结果
                        cv2.imwrite(output_path, result_image)
                    
                    # 简化返回的分割数据（避免返回大数组）
                    segmentation_info = {
                        'unique_classes': [int(c) for c in np.unique(class_map)],
                        'mask_shape': mask.shape
                    }
                    
                    results[op] = {
                        'success': True,
                        'segmentation': segmentation_info,
                        'result_image': image_to_base64(output_path) if os.path.exists(output_path) else None
                    }
                else:
                    results[op] = {
                        'success': False,
                        'error': '缺少必要的配置文件或类别文件'
                    }
            
            else:
                results[op] = {
                    'success': False,
                    'error': f'不支持的操作: {op}'
                }
        
        # 返回处理结果
        return jsonify({
            'success': True,
            'input_image': input_path,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # 可选：清理上传的临时文件
        # os.remove(input_path)
        pass

@api_bp.route('/add_face', methods=['POST'])
def api_add_face():
    """添加人脸到数据库"""
    # 检查是否有文件或Base64数据
    if 'image' in request.files:
        file = request.files['image']
        input_path = save_uploaded_file(file)
    elif request.json and 'image_base64' in request.json:
        input_path = save_base64_image(request.json['image_base64'])
    else:
        return jsonify({'error': '未找到图像文件或Base64数据'}), 400
    
    if not input_path:
        return jsonify({'error': '无效的图像格式'}), 400
    
    # 获取人名
    name = request.form.get('name') if request.form else None
    if request.json:
        name = request.json.get('name')
    
    if not name:
        return jsonify({'error': '未提供人名'}), 400
    
    try:
        # 创建人脸识别器实例
        # recognizer = FaceRecognizer()
        
        # 加载图像
        image = cv2.imread(input_path)
        
        # 添加人脸
        # success = recognizer.add_face(image, name)
        
        return jsonify({
            'success': False,
            'error': '人脸识别功能暂时不可用'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/add_image_to_db', methods=['POST'])
def api_add_image_to_db():
    """添加图像到图像检索数据库"""
    # 检查是否有文件或Base64数据
    if 'image' in request.files:
        file = request.files['image']
        input_path = save_uploaded_file(file)
    elif request.json and 'image_base64' in request.json:
        input_path = save_base64_image(request.json['image_base64'])
    else:
        return jsonify({'error': '未找到图像文件或Base64数据'}), 400
    
    if not input_path:
        return jsonify({'error': '无效的图像格式'}), 400
    
    # 获取数据库路径
    db_path = request.form.get('db_path') if request.form else None
    if request.json:
        db_path = request.json.get('db_path')
    
    if not db_path:
        db_path = 'models/image_db_sift.pkl'  # 默认数据库路径
    
    # 获取描述符类型
    descriptor = request.form.get('descriptor', 'sift') if request.form else 'sift'
    if request.json:
        descriptor = request.json.get('descriptor', 'sift')
    
    try:
        # success = add_image_to_db(input_path, db_path, descriptor=descriptor)
        return jsonify({
            'success': False,
            'error': '图像检索功能暂时不可用'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/docs', methods=['GET'])
def api_docs():
    """返回API文档"""
    docs = {
        'title': 'OpenCV图像处理与计算机视觉API',
        'version': '1.0',
        'description': '提供图像处理、计算机视觉和深度学习功能的RESTful API',
        'endpoints': [
            {
                'path': '/api/status',
                'method': 'GET',
                'description': '检查API服务状态',
                'parameters': []
            },
            {
                'path': '/api/process',
                'method': 'POST',
                'description': '处理图像',
                'parameters': [
                    {'name': 'image', 'type': 'file', 'required': False, 'description': '图像文件'},
                    {'name': 'image_base64', 'type': 'string', 'required': False, 'description': 'Base64编码的图像数据'},
                    {'name': 'operations', 'type': 'string', 'required': False, 'description': '要执行的操作，用逗号分隔'}
                ]
            },
            {
                'path': '/api/add_face',
                'method': 'POST',
                'description': '添加人脸到数据库',
                'parameters': [
                    {'name': 'image', 'type': 'file', 'required': False, 'description': '图像文件'},
                    {'name': 'image_base64', 'type': 'string', 'required': False, 'description': 'Base64编码的图像数据'},
                    {'name': 'name', 'type': 'string', 'required': True, 'description': '人名'}
                ]
            },
            {
                'path': '/api/add_image_to_db',
                'method': 'POST',
                'description': '添加图像到图像检索数据库',
                'parameters': [
                    {'name': 'image', 'type': 'file', 'required': False, 'description': '图像文件'},
                    {'name': 'image_base64', 'type': 'string', 'required': False, 'description': 'Base64编码的图像数据'},
                    {'name': 'db_path', 'type': 'string', 'required': False, 'description': '数据库路径'},
                    {'name': 'descriptor', 'type': 'string', 'required': False, 'description': '描述符类型：sift, orb, brisk, akaze'}
                ]
            },
            {
                'path': '/api/docs',
                'method': 'GET',
                'description': '获取API文档',
                'parameters': []
            }
        ],
        'operations': [
            {'name': 'basic', 'description': '基本图像处理'},
            {'name': 'face_detect', 'description': '人脸检测'},
            {'name': 'face_recognize', 'description': '人脸识别'},
            {'name': 'edge', 'description': '边缘检测'},
            {'name': 'color', 'description': '颜色分析'},
            {'name': 'object_detect', 'description': '物体检测'},
            {'name': 'image_retrieve', 'description': '图像检索'},
            {'name': 'deep_classify', 'description': '深度学习图像分类'},
            {'name': 'deep_detect', 'description': '深度学习物体检测'},
            {'name': 'segment', 'description': '图像分割'}
        ]
    }
    return jsonify(docs)

# 图像检索API
@api_bp.route('/retrieval', methods=['POST'])
def image_retrieval_api():
    """图像检索API端点
    
    请求参数:
    - image: 上传的图像文件或base64编码的图像
    - feature_type: 特征类型 (sift, orb, deep, color_hist)
    - max_results: 最大结果数
    - similarity_threshold: 相似度阈值
    - show_matches: 是否显示特征匹配
    
    返回:
    - 检索结果列表，包含相似图像和相似度得分
    """
    try:
        # 获取查询图像
        image = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                # 从文件读取图像
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'image_base64' in request.form:
            # 从base64解码图像
            image_b64 = request.form['image_base64']
            image = base64_to_image(image_b64)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            })
        
        # 获取检索参数
        feature_type = request.form.get('feature_type', 'sift')
        max_results = int(request.form.get('max_results', 10))
        similarity_threshold = float(request.form.get('similarity_threshold', 0.6))
        show_matches = request.form.get('show_matches', '1') == '1'
        
        # 执行图像检索
        results = retrieval_system.retrieve_similar_images(
            image, 
            feature_type=feature_type,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        # 如果不需要显示匹配，移除matches_image字段以减小响应大小
        if not show_matches:
            for result in results:
                if 'matches_image' in result:
                    del result['matches_image']
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/retrieval/add_to_db', methods=['POST'])
def add_to_db():
    """添加图像到数据库的API端点
    
    请求参数:
    - images: 上传的图像文件（可多个）
    
    返回:
    - 添加成功的图像数量
    """
    try:
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            })
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({
                'success': False,
                'error': 'No images selected'
            })
        
        added_count = 0
        for file in files:
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 读取图像
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # 添加到数据库
                    image_id = retrieval_system.add_image_to_db(
                        image, 
                        filename=file.filename,
                        extract_all_features=True
                    )
                    
                    if image_id:
                        added_count += 1
        
        return jsonify({
            'success': True,
            'added_count': added_count
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/retrieval/list_db', methods=['GET'])
def list_db():
    """列出数据库中的所有图像API端点
    
    返回:
    - 图像列表，包含ID、文件名和缩略图
    """
    try:
        images = retrieval_system.get_all_images(include_thumbnails=True)
        
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/retrieval/clear_db', methods=['POST'])
def clear_db():
    """清空图像数据库API端点"""
    try:
        success = retrieval_system.clear_database()
        
        return jsonify({
            'success': success
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/retrieval/rebuild_index', methods=['POST'])
def rebuild_index():
    """重建特征索引API端点"""
    try:
        success = retrieval_system.rebuild_index()
        
        return jsonify({
            'success': success
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 修改main函数
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='OpenCV图像处理与计算机视觉API服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    print(f"启动API服务于 http://{args.host}:{args.port}")
    print("API文档: http://localhost:5000/api/docs")
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 