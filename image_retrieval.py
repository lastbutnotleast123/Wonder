#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import pickle
from typing import List, Tuple, Dict, Optional, Union
import time
import glob
import uuid
import base64
from pathlib import Path
import shutil
from datetime import datetime
import logging

import image_utils

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_retrieval")

class ImageDescriptor:
    """图像描述符提取类"""
    
    def __init__(self, method: str = "sift"):
        """
        初始化图像描述符提取器
        
        Args:
            method: 描述符提取方法，可选 'sift', 'orb', 'brisk', 'akaze'
        """
        self.method = method.lower()
        
        # 创建描述符提取器
        if self.method == "sift":
            self.detector = cv2.SIFT_create()
        elif self.method == "orb":
            self.detector = cv2.ORB_create()
        elif self.method == "brisk":
            self.detector = cv2.BRISK_create()
        elif self.method == "akaze":
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"不支持的描述符方法: {method}")
        
        # 创建特征匹配器
        if self.method == "sift" or self.method == "akaze":
            # For SIFT or AKAZE (浮点描述符)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            # For ORB or BRISK (二进制描述符)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        提取图像特征点和描述符
        
        Args:
            image: 输入图像
            
        Returns:
            (关键点列表, 描述符数组)
        """
        # 转换为灰度图像
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测关键点并计算描述符
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_thresh: float = 0.75) -> List[cv2.DMatch]:
        """
        匹配两组特征描述符
        
        Args:
            desc1: 第一组描述符
            desc2: 第二组描述符
            ratio_thresh: Lowe比率测试阈值
            
        Returns:
            匹配列表
        """
        # 检查描述符是否为空
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # 使用KNN匹配找到最佳2个匹配
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # 应用Lowe比率测试筛选出好的匹配
        good_matches = []
        for match in matches:
            if len(match) == 2:  # 确保找到了2个匹配
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        计算两组描述符之间的相似度
        
        Args:
            desc1: 第一组描述符
            desc2: 第二组描述符
            
        Returns:
            相似度分数 (0-1之间，1表示完全匹配)
        """
        # 获取匹配
        matches = self.match_features(desc1, desc2)
        
        # 计算相似度
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        # 相似度定义为好匹配数量与最小描述符数量的比率
        min_desc_count = min(len(desc1), len(desc2))
        similarity = len(matches) / min_desc_count if min_desc_count > 0 else 0
        
        return min(similarity, 1.0)  # 限制最大为1.0

class ImageDatabase:
    """图像数据库类，用于存储和检索图像"""
    
    def __init__(self, descriptor_method: str = "sift", 
                db_path: Optional[str] = None):
        """
        初始化图像数据库
        
        Args:
            descriptor_method: 描述符提取方法
            db_path: 数据库文件路径
        """
        self.descriptor = ImageDescriptor(descriptor_method)
        self.db_path = db_path or f"models/image_db_{descriptor_method}.pkl"
        self.image_descriptors = {}  # {image_path: (descriptors, metadata)}
        
        # 如果数据库文件存在，则加载
        if os.path.exists(self.db_path):
            self.load_database()
    
    def load_database(self) -> None:
        """加载图像数据库"""
        try:
            with open(self.db_path, "rb") as f:
                self.image_descriptors = pickle.load(f)
            print(f"已加载 {len(self.image_descriptors)} 张图像的描述符")
        except Exception as e:
            print(f"加载图像数据库时出错: {e}")
            self.image_descriptors = {}
    
    def save_database(self) -> None:
        """保存图像数据库"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 保存数据库
        with open(self.db_path, "wb") as f:
            pickle.dump(self.image_descriptors, f)
        print(f"已保存 {len(self.image_descriptors)} 张图像的描述符到 {self.db_path}")
    
    def add_image(self, image_path: str, metadata: Optional[Dict] = None) -> bool:
        """
        向数据库添加图像
        
        Args:
            image_path: 图像文件路径
            metadata: 与图像关联的元数据
            
        Returns:
            是否成功添加
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return False
        
        try:
            # 加载图像
            image = image_utils.load_image(image_path)
            
            # 提取特征
            _, descriptors = self.descriptor.extract_features(image)
            
            # 如果未能提取特征，则返回失败
            if descriptors is None:
                print(f"无法从图像中提取特征: {image_path}")
                return False
            
            # 存储描述符和元数据
            self.image_descriptors[image_path] = (descriptors, metadata or {})
            
            # 保存数据库
            self.save_database()
            
            return True
        except Exception as e:
            print(f"添加图像时出错: {e}")
            return False
    
    def batch_add_images(self, image_pattern: str) -> int:
        """
        批量添加图像
        
        Args:
            image_pattern: 图像文件通配符模式
            
        Returns:
            成功添加的图像数量
        """
        # 查找匹配的图像文件
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            print(f"没有找到匹配的图像: {image_pattern}")
            return 0
        
        # 批量添加图像
        success_count = 0
        for image_path in image_files:
            if self.add_image(image_path):
                success_count += 1
                print(f"已添加 ({success_count}/{len(image_files)}): {image_path}")
        
        return success_count
    
    def search_image(self, query_image: np.ndarray, 
                    top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似图像
        
        Args:
            query_image: 查询图像
            top_k: 返回的最相似图像数量
            
        Returns:
            相似图像列表 [(image_path, similarity_score, metadata), ...]
        """
        # 如果数据库为空，返回空列表
        if not self.image_descriptors:
            print("图像数据库为空")
            return []
        
        # 提取查询图像的特征
        _, query_desc = self.descriptor.extract_features(query_image)
        
        # 如果无法提取特征，返回空列表
        if query_desc is None:
            print("无法从查询图像中提取特征")
            return []
        
        # 计算与数据库中所有图像的相似度
        similarities = []
        for image_path, (desc, metadata) in self.image_descriptors.items():
            similarity = self.descriptor.compute_similarity(query_desc, desc)
            similarities.append((image_path, similarity, metadata))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k个结果
        return similarities[:top_k]

def visualize_matches(img1: np.ndarray, img2: np.ndarray, 
                     kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                     matches: List[cv2.DMatch], max_matches: int = 50) -> np.ndarray:
    """
    可视化两张图像之间的特征匹配
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        kp1: 第一张图像的关键点
        kp2: 第二张图像的关键点
        matches: 匹配列表
        max_matches: 最多显示的匹配数量
        
    Returns:
        匹配可视化图像
    """
    # 限制匹配数量
    matches = matches[:max_matches] if len(matches) > max_matches else matches
    
    # 绘制匹配
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return result

class ImageRetrieval:
    def __init__(self, db_dir='image_db', features_dir='image_features'):
        """初始化图像检索系统
        
        Args:
            db_dir: 图像数据库目录
            features_dir: 特征存储目录
        """
        self.db_dir = db_dir
        self.features_dir = features_dir
        
        # 创建必要的目录
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(os.path.join(features_dir, 'sift'), exist_ok=True)
        os.makedirs(os.path.join(features_dir, 'orb'), exist_ok=True)
        os.makedirs(os.path.join(features_dir, 'deep'), exist_ok=True)
        os.makedirs(os.path.join(features_dir, 'color_hist'), exist_ok=True)
        
        # 加载索引（如果存在）
        self.index = self._load_index()
        
        # 初始化特征提取器
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # 用于深度特征的模型
        self.deep_model = None
        
        logger.info(f"图像检索系统初始化完成，数据库位置: {db_dir}")
    
    def _load_index(self):
        """加载索引文件"""
        index_path = os.path.join(self.features_dir, 'index.pkl')
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"加载索引文件失败: {e}")
                return {}
        return {}
    
    def _save_index(self):
        """保存索引文件"""
        index_path = os.path.join(self.features_dir, 'index.pkl')
        try:
            with open(index_path, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}")
    
    def extract_features(self, image, feature_type='sift'):
        """从图像中提取特征
        
        Args:
            image: OpenCV格式的图像
            feature_type: 特征类型 ('sift', 'orb', 'deep', 'color_hist')
            
        Returns:
            提取的特征
        """
        if feature_type == 'sift':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            return {'keypoints': keypoints, 'descriptors': descriptors}
        
        elif feature_type == 'orb':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            return {'keypoints': keypoints, 'descriptors': descriptors}
        
        elif feature_type == 'deep':
            # 这里应该集成深度学习模型进行特征提取
            # 由于需要外部依赖，这里简单模拟
            if self.deep_model is None:
                # 简单模拟，实际应该使用预训练CNN
                resized = cv2.resize(image, (224, 224))
                features = np.mean(resized.reshape(-1, 3), axis=0)
                return {'features': features}
            else:
                # 实际应该使用深度学习模型提取特征
                pass
        
        elif feature_type == 'color_hist':
            # 计算颜色直方图
            hist_b = cv2.calcHist([image], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [64], [0, 256])
            
            # 归一化
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            
            # 组合直方图
            hist = np.concatenate([hist_b, hist_g, hist_r])
            return {'histogram': hist}
        
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
    
    def add_image_to_db(self, image, filename=None, extract_all_features=True):
        """添加图像到数据库
        
        Args:
            image: OpenCV格式的图像或图像文件路径
            filename: 图像文件名，如果为None则自动生成
            extract_all_features: 是否提取所有类型的特征
            
        Returns:
            image_id: 添加的图像ID
        """
        # 如果image是字符串，假定它是文件路径并加载图像
        if isinstance(image, str):
            try:
                image = cv2.imread(image)
            except Exception as e:
                logger.error(f"加载图像失败: {e}")
                return None
        
        if image is None or image.size == 0:
            logger.error("无效的图像数据")
            return None
        
        # 生成唯一ID用于存储
        image_id = str(uuid.uuid4())
        
        # 如果文件名未提供，则使用ID作为文件名
        if filename is None:
            filename = f"{image_id}.jpg"
        
        # 存储图像
        image_path = os.path.join(self.db_dir, f"{image_id}.jpg")
        cv2.imwrite(image_path, image)
        
        # 记录索引信息
        self.index[image_id] = {
            'filename': filename,
            'path': image_path,
            'added': datetime.now().isoformat(),
            'features': {}
        }
        
        # 提取并存储特征
        feature_types = ['sift', 'orb', 'deep', 'color_hist'] if extract_all_features else ['sift']
        
        for feature_type in feature_types:
            try:
                features = self.extract_features(image, feature_type)
                
                # 存储特征
                feature_path = os.path.join(self.features_dir, feature_type, f"{image_id}.pkl")
                with open(feature_path, 'wb') as f:
                    pickle.dump(features, f)
                
                # 更新索引
                self.index[image_id]['features'][feature_type] = feature_path
                
                logger.info(f"为图像 {image_id} 提取并存储了 {feature_type} 特征")
            except Exception as e:
                logger.error(f"提取 {feature_type} 特征失败: {e}")
        
        # 保存更新后的索引
        self._save_index()
        
        return image_id
    
    def retrieve_similar_images(self, query_image, feature_type='sift', max_results=10, similarity_threshold=0.6):
        """检索与查询图像相似的图像
        
        Args:
            query_image: 查询图像（OpenCV格式）
            feature_type: 用于检索的特征类型
            max_results: 返回的最大结果数
            similarity_threshold: 相似度阈值
            
        Returns:
            检索结果列表，每个结果是字典，包含image_id, similarity, matches等信息
        """
        if not self.index:
            logger.warning("图像数据库为空")
            return []
        
        # 提取查询图像的特征
        query_features = self.extract_features(query_image, feature_type)
        
        results = []
        
        # 颜色直方图和深度特征使用不同的匹配方法
        if feature_type in ['color_hist', 'deep']:
            query_vector = query_features['histogram'] if feature_type == 'color_hist' else query_features['features']
            
            for image_id, info in self.index.items():
                if feature_type not in info['features']:
                    continue
                
                try:
                    # 加载存储的特征
                    with open(info['features'][feature_type], 'rb') as f:
                        db_features = pickle.load(f)
                    
                    # 计算相似度 (使用相关系数或欧氏距离的倒数)
                    db_vector = db_features['histogram'] if feature_type == 'color_hist' else db_features['features']
                    
                    if feature_type == 'color_hist':
                        # 使用相关系数计算相似度 (范围为 [-1, 1]，转换为 [0, 1])
                        similarity = (cv2.compareHist(query_vector, db_vector, cv2.HISTCMP_CORREL) + 1) / 2
                    else:
                        # 欧氏距离的归一化倒数作为相似度
                        dist = np.linalg.norm(query_vector - db_vector)
                        similarity = 1 / (1 + dist)
                    
                    if similarity >= similarity_threshold:
                        # 加载图像用于显示
                        image = cv2.imread(info['path'])
                        
                        results.append({
                            'image_id': image_id,
                            'similarity': float(similarity),
                            'filename': info['filename'],
                            'image': self._encode_image(image),
                            'matches_count': 0
                        })
                
                except Exception as e:
                    logger.error(f"处理图像 {image_id} 失败: {e}")
        
        # SIFT 和 ORB 使用特征点匹配
        else:
            # 获取查询图像的描述符
            query_descriptors = query_features.get('descriptors')
            if query_descriptors is None or len(query_descriptors) == 0:
                logger.warning("查询图像中未检测到特征点")
                return []
            
            # 创建FLANN匹配器或暴力匹配器
            if feature_type == 'sift':
                # FLANN匹配器，适用于SIFT/SURF
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # 暴力匹配器，适用于ORB等二进制特征
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            for image_id, info in self.index.items():
                if feature_type not in info['features']:
                    continue
                
                try:
                    # 加载存储的特征
                    with open(info['features'][feature_type], 'rb') as f:
                        db_features = pickle.load(f)
                    
                    db_descriptors = db_features.get('descriptors')
                    if db_descriptors is None or len(db_descriptors) == 0:
                        continue
                    
                    # 匹配描述符
                    if feature_type == 'sift':
                        # knnMatch对于FLANN
                        matches = matcher.knnMatch(query_descriptors, db_descriptors, k=2)
                        
                        # Lowe's比率测试
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                    else:
                        # 使用BFMatcher的match方法
                        matches = matcher.match(query_descriptors, db_descriptors)
                        
                        # 按距离排序
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        # 取前30%作为好的匹配
                        good_matches = matches[:int(len(matches) * 0.3)]
                    
                    # 计算相似度得分
                    if len(good_matches) > 0:
                        # 归一化得分 (0-1范围)
                        similarity = len(good_matches) / min(len(query_descriptors), len(db_descriptors))
                        
                        if similarity >= similarity_threshold:
                            # 加载图像用于显示
                            db_image = cv2.imread(info['path'])
                            
                            # 创建匹配图像
                            matches_image = None
                            if len(good_matches) >= 4:  # 需要至少4个点用于透视变换
                                try:
                                    # 重新提取特征点（因为序列化不会保存完整的KeyPoint对象）
                                    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
                                    db_gray = cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY)
                                    
                                    if feature_type == 'sift':
                                        query_kp, _ = self.sift.detectAndCompute(query_gray, None)
                                        db_kp, _ = self.sift.detectAndCompute(db_gray, None)
                                    else:
                                        query_kp, _ = self.orb.detectAndCompute(query_gray, None)
                                        db_kp, _ = self.orb.detectAndCompute(db_gray, None)
                                    
                                    # 绘制匹配
                                    matches_image = cv2.drawMatches(
                                        query_image, query_kp, 
                                        db_image, db_kp, 
                                        good_matches[:50],  # 限制为最多50个匹配点，避免图像过于混乱
                                        None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                    )
                                except Exception as e:
                                    logger.error(f"绘制匹配失败: {e}")
                            
                            results.append({
                                'image_id': image_id,
                                'similarity': float(similarity),
                                'filename': info['filename'],
                                'image': self._encode_image(db_image),
                                'matches_count': len(good_matches),
                                'matches_image': self._encode_image(matches_image) if matches_image is not None else None
                            })
                
                except Exception as e:
                    logger.error(f"处理图像 {image_id} 失败: {e}")
        
        # 按相似度排序
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # 限制结果数量
        return results[:max_results]
    
    def clear_database(self):
        """清空图像数据库和特征库"""
        try:
            # 清空数据库目录
            shutil.rmtree(self.db_dir)
            os.makedirs(self.db_dir, exist_ok=True)
            
            # 清空特征目录
            for feature_type in ['sift', 'orb', 'deep', 'color_hist']:
                feature_dir = os.path.join(self.features_dir, feature_type)
                shutil.rmtree(feature_dir)
                os.makedirs(feature_dir, exist_ok=True)
            
            # 重置索引
            self.index = {}
            self._save_index()
            
            logger.info("成功清空图像数据库")
            return True
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def rebuild_index(self):
        """重建特征索引"""
        try:
            # 保存旧索引文件名列表
            old_filenames = {image_id: info['filename'] for image_id, info in self.index.items()}
            
            # 重新初始化索引
            self.index = {}
            
            # 遍历数据库文件夹
            for filename in os.listdir(self.db_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_id = os.path.splitext(filename)[0]
                    image_path = os.path.join(self.db_dir, filename)
                    
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # 获取原始文件名（如果存在）
                    original_filename = old_filenames.get(image_id, filename)
                    
                    # 重新添加到索引
                    self.index[image_id] = {
                        'filename': original_filename,
                        'path': image_path,
                        'added': datetime.now().isoformat(),
                        'features': {}
                    }
                    
                    # 提取所有特征
                    for feature_type in ['sift', 'orb', 'deep', 'color_hist']:
                        try:
                            features = self.extract_features(image, feature_type)
                            
                            # 存储特征
                            feature_path = os.path.join(self.features_dir, feature_type, f"{image_id}.pkl")
                            with open(feature_path, 'wb') as f:
                                pickle.dump(features, f)
                            
                            # 更新索引
                            self.index[image_id]['features'][feature_type] = feature_path
                        except Exception as e:
                            logger.error(f"重建 {feature_type} 特征失败: {e}")
            
            # 保存更新后的索引
            self._save_index()
            
            logger.info(f"成功重建索引，共 {len(self.index)} 张图像")
            return True
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False
    
    def get_all_images(self, include_thumbnails=True):
        """获取数据库中的所有图像
        
        Args:
            include_thumbnails: 是否包含缩略图
            
        Returns:
            图像列表，每个元素是字典，包含image_id、filename及可选的thumbnail
        """
        images = []
        
        for image_id, info in self.index.items():
            image_info = {
                'image_id': image_id,
                'filename': info['filename'],
                'added': info['added']
            }
            
            if include_thumbnails:
                try:
                    # 读取图像并生成缩略图
                    image = cv2.imread(info['path'])
                    if image is not None:
                        # 创建缩略图
                        height, width = image.shape[:2]
                        max_dim = 150
                        scale = max_dim / max(height, width)
                        thumbnail = cv2.resize(image, (int(width * scale), int(height * scale)))
                        
                        # 编码为base64
                        image_info['thumbnail'] = self._encode_image(thumbnail)
                except Exception as e:
                    logger.error(f"生成缩略图失败: {e}")
            
            images.append(image_info)
        
        # 按添加时间排序
        images.sort(key=lambda x: x.get('added', ''), reverse=True)
        
        return images
    
    @staticmethod
    def _encode_image(image):
        """将OpenCV图像编码为base64字符串
        
        Args:
            image: OpenCV格式图像
            
        Returns:
            base64编码的字符串
        """
        if image is None:
            return None
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image)
        # 转换为base64字符串
        return base64.b64encode(buffer).decode('utf-8')


# 创建单例实例
retrieval_system = ImageRetrieval()

def initialize():
    """初始化图像检索系统"""
    global retrieval_system
    retrieval_system = ImageRetrieval()
    return retrieval_system

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="图像检索程序")
    parser.add_argument("--mode", choices=["add", "search", "batch_add"], default="search",
                       help="操作模式：添加图像(add)、搜索图像(search)或批量添加(batch_add)")
    parser.add_argument("--image", help="输入图像路径")
    parser.add_argument("--pattern", help="批量添加模式下的图像通配符模式")
    parser.add_argument("--method", choices=["sift", "orb", "brisk", "akaze"], default="sift",
                       help="特征描述符方法")
    parser.add_argument("--top_k", type=int, default=5, help="返回的最相似图像数量")
    parser.add_argument("--display", action="store_true", help="是否显示结果")
    parser.add_argument("--output", help="输出图像路径")
    args = parser.parse_args()
    
    # 创建图像数据库
    db = ImageDatabase(args.method)
    
    if args.mode == "add":
        # 添加模式
        if not args.image:
            print("错误：添加图像时必须提供图像路径")
            return
        
        # 添加图像到数据库
        success = db.add_image(args.image)
        
        if success:
            print(f"已成功添加图像: {args.image}")
        else:
            print(f"添加图像失败: {args.image}")
    
    elif args.mode == "batch_add":
        # 批量添加模式
        if not args.pattern:
            print("错误：批量添加模式下必须提供图像通配符模式")
            return
        
        # 批量添加图像
        count = db.batch_add_images(args.pattern)
        print(f"已成功添加 {count} 张图像")
    
    else:  # search 模式
        # 搜索模式
        if not args.image:
            print("错误：搜索图像时必须提供查询图像路径")
            return
        
        # 加载查询图像
        query_image = image_utils.load_image(args.image)
        
        # 开始计时
        start_time = time.time()
        
        # 搜索相似图像
        results = db.search_image(query_image, args.top_k)
        
        # 计算用时
        elapsed_time = time.time() - start_time
        
        # 打印搜索结果
        print(f"搜索完成，共找到 {len(results)} 个相似图像，用时 {elapsed_time:.2f} 秒")
        
        if results:
            # 如果需要显示结果
            if args.display:
                # 提取查询图像的特征
                descriptor = ImageDescriptor(args.method)
                query_kp, query_desc = descriptor.extract_features(query_image)
                
                # 显示最相似的图像及其匹配
                for i, (image_path, similarity, _) in enumerate(results):
                    # 加载目标图像
                    target_image = image_utils.load_image(image_path)
                    
                    # 提取目标图像的特征
                    target_kp, target_desc = descriptor.extract_features(target_image)
                    
                    # 匹配特征
                    matches = descriptor.match_features(query_desc, target_desc)
                    
                    # 显示匹配结果
                    match_img = visualize_matches(query_image, target_image, 
                                                 query_kp, target_kp, matches)
                    
                    # 显示图像
                    title = f"匹配结果 #{i+1}: {os.path.basename(image_path)} (相似度: {similarity:.2f})"
                    image_utils.display_image(match_img, title)
                    
                    # 如果指定了输出路径，且是第一个结果，则保存
                    if args.output and i == 0:
                        image_utils.save_image(match_img, args.output)
                        print(f"已保存匹配结果到: {args.output}")
            
            # 打印相似图像
            for i, (image_path, similarity, metadata) in enumerate(results):
                print(f"#{i+1}: {image_path} (相似度: {similarity:.2f})")
        else:
            print("未找到相似图像")

if __name__ == "__main__":
    main() 