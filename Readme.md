# 基于OpenCV的图像处理与计算机视觉项目使用说明

## 项目简介

这是一个基于Python和OpenCV库实现的综合图像处理与计算机视觉项目，包含丰富的图像处理和识别功能，如基本图像处理、人脸检测与识别、物体检测与追踪、图像检索、边缘检测、颜色分析以及深度学习应用等。

## 功能特点

- **基本图像处理**：调整大小、旋转、滤波、颜色转换等
- **人脸检测与识别**：检测图像中的人脸并标记、识别已知人脸
- **图像检索**：基于图像描述符的图像搜索和匹配
- **物体检测与识别**：检测和识别图像中的常见物体
- **目标跟踪**：视频中的物体跟踪
- **颜色检测**：提取图像中的主要颜色
- **边缘检测**：检测并突出显示图像中的边缘
- **OpenCV深度学习**：使用预训练模型进行图像分类、物体检测和语义分割

## 环境配置

1. 确保已安装Python 3.7或更高版本
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

3. 对于物体检测功能，需要下载预训练模型：
   - 访问 https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
   - 下载 SSD MobileNet V2 模型文件
   - 将模型文件放置在 `models` 目录下：
     - `models/ssd_mobilenet_v2_coco_2018_03_29.pb`
     - `models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt`

4. 对于深度学习功能，需要下载适当的预训练模型和配置文件，放置在 `models` 目录下。

## 使用方法

### 主程序

主程序集成了所有功能，可以处理图像和视频：

```bash
# 图像处理模式
python main.py --mode image --input 图像路径 [--output 输出目录] [--operations 操作列表] [--display]

# 视频处理模式
python main.py --mode video --input 视频路径 [--output 输出目录] --operations 操作 [--display]
```

参数说明：
- `--input`：输入图像或视频路径（必需）
- `--output`：输出目录，默认为 `output`
- `--mode`：处理模式，可选值为 `image` 或 `video`，默认为 `image`
- `--operations`：要应用的操作列表，可选值包括：
  - `all`：应用所有基本操作
  - `face_detect`：人脸检测
  - `face_recognize`：人脸识别
  - `edge`：边缘检测
  - `color`：颜色分析
  - `object_detect`：物体检测
  - `image_retrieve`：图像检索
  - `deep_classify`：深度学习图像分类
  - `deep_detect`：深度学习物体检测
  - `segment`：图像分割
  - `object_track`：视频目标跟踪（仅视频模式）
  - `face_track`：视频人脸跟踪（仅视频模式）
- `--display`：是否显示处理结果

高级参数：
- `--db_path`：图像检索数据库路径
- `--model_path`：深度学习模型路径
- `--config_path`：深度学习模型配置路径
- `--classes_file`：类别文件路径
- `--framework`：深度学习框架，可选值为 `tensorflow`、`caffe`、`darknet`、`torch`，默认为 `tensorflow`
- `--confidence`：置信度阈值，默认为 0.5
- `--top_k`：返回的结果数量，默认为 5
- `--tracker_type`：目标跟踪器类型，可选值为 `csrt`、`kcf`、`boosting`、`mil`、`tld`、`medianflow`、`mosse`，默认为 `csrt`

示例：
```bash
# 对图像应用人脸检测和边缘检测
python main.py --mode image --input images/sample.jpg --operations face_detect edge --display

# 对视频进行目标跟踪
python main.py --mode video --input videos/sample.mp4 --operations object_track --tracker_type csrt --display
```

### 独立功能模块

每个功能也可以作为独立模块使用：

#### 人脸检测

```bash
python face_detection.py --image 图像路径 [--output 输出路径] [--eyes] [--display]
```

#### 人脸识别

```bash
python face_recognition_module.py --mode recognize --image 图像路径 [--output 输出路径] [--display]

# 添加新人脸
python face_recognition_module.py --mode add --image 图像路径 --name "人名"
```

#### 图像检索

```bash
# 添加图像到数据库
python image_retrieval.py --mode add --image 图像路径

# 批量添加图像
python image_retrieval.py --mode batch_add --pattern "images/*.jpg"

# 搜索相似图像
python image_retrieval.py --mode search --image 图像路径 [--top_k 5] [--display]
```

#### 目标跟踪

```bash
python object_tracking.py --video 视频路径 [--output 输出路径] [--tracker csrt] [--display]
```

#### 深度学习应用

```bash
# 图像分类
python deep_learning.py --mode classify --image 图像路径 --model 模型路径 [--config 配置路径] --classes 类别文件 [--display]

# 物体检测
python deep_learning.py --mode detect --image 图像路径 --model 模型路径 --config 配置路径 [--classes 类别文件] [--display]

# 图像分割
python deep_learning.py --mode segment --image 图像路径 --model 模型路径 --config 配置路径 [--classes 类别文件] [--display]
```

## 各功能模块详细说明

### 1. 基本图像处理 (image_utils.py)

提供基本的图像处理功能，包括：
- 加载和保存图像
- 调整图像大小
- 转换为灰度图像
- 应用高斯模糊
- 绘制矩形和文本
- 等等

### 2. 人脸检测 (face_detection.py)

- 使用Haar级联分类器检测人脸
- 可选择性地检测人脸中的眼睛
- 在图像上标记检测结果

### 3. 人脸识别 (face_recognition_module.py)

- 基于开源的face_recognition库实现
- 支持添加新人脸
- 能够识别已知人脸并标记名称
- 使用人脸编码保存人脸特征

### 4. 图像检索 (image_retrieval.py)

- 使用SIFT、ORB、BRISK或AKAZE特征描述符
- 构建图像数据库
- 支持相似图像搜索
- 可视化特征匹配结果

### 5. 物体检测 (object_detection.py)

- 使用预训练的SSD MobileNet模型
- 支持COCO数据集中的80多种常见物体
- 在图像上标记检测结果

### 6. 目标跟踪 (object_tracking.py)

- 支持多种跟踪算法（CSRT、KCF、BOOSTING等）
- 可交互式选择跟踪目标
- 支持多目标同时跟踪
- 实时显示跟踪结果

### 7. 边缘检测 (edge_detection.py)

- 支持Canny、Sobel和Laplacian边缘检测算法
- 可调整检测参数
- 支持将边缘叠加到原始图像上

### 8. 颜色检测 (color_detection.py)

- 提取图像中的主要颜色
- 支持HSV颜色空间的特定颜色范围检测
- 生成颜色直方图
- 可视化颜色分布

### 9. 深度学习应用 (deep_learning.py)

- 支持多种深度学习框架（TensorFlow、Caffe、Darknet、PyTorch）
- 图像分类：识别图像中的主要对象
- 物体检测：定位和识别图像中的多个对象
- 语义分割：对图像中的每个像素进行分类

## 示例

1. 将示例图像放在 `images` 目录下
2. 将示例视频放在 `videos` 目录下
3. 运行以下命令：

```bash
# 基本图像处理
python main.py --mode image --input images/sample.jpg --operations all --display

# 人脸识别
python main.py --mode image --input images/faces.jpg --operations face_recognize --display

# 图像检索（需要先添加图像到数据库）
python image_retrieval.py --mode batch_add --pattern "images/*.jpg"
python main.py --mode image --input images/query.jpg --operations image_retrieve --db_path models/image_db_sift.pkl --display

# 视频目标跟踪
python main.py --mode video --input videos/sample.mp4 --operations object_track --display
```

## 注意事项

- 物体检测功能需要下载预训练模型
- 深度学习功能需要相应的模型和配置文件
- 显示结果需要图形界面支持
- 处理大尺寸图像或视频可能需要较长时间
- 人脸识别需要先添加人脸到数据库
- 图像检索需要先构建图像数据库 