#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import time
from typing import List, Tuple, Dict, Optional, Union

import image_utils

class ObjectTracker:
    """目标跟踪器类"""
    
    TRACKER_TYPES = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.legacy.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create
    }
    
    def __init__(self, tracker_type: str = "csrt"):
        """
        初始化目标跟踪器
        
        Args:
            tracker_type: 跟踪器类型，可选值为 'csrt', 'kcf', 'boosting', 'mil', 'tld', 'medianflow', 'mosse'
        """
        if tracker_type.lower() not in self.TRACKER_TYPES:
            raise ValueError(f"不支持的跟踪器类型: {tracker_type}，可选值为: {list(self.TRACKER_TYPES.keys())}")
        
        self.tracker_type = tracker_type.lower()
        self.tracker = None
        self.create_new_tracker()
    
    def create_new_tracker(self) -> None:
        """创建新的跟踪器实例"""
        self.tracker = self.TRACKER_TYPES[self.tracker_type]()
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        初始化跟踪器
        
        Args:
            frame: 视频帧
            bbox: 目标边界框 (x, y, width, height)
            
        Returns:
            是否成功初始化
        """
        return self.tracker.init(frame, bbox)
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """
        更新跟踪器
        
        Args:
            frame: 当前视频帧
            
        Returns:
            (是否成功跟踪, 边界框 (x, y, width, height))
        """
        return self.tracker.update(frame)
    
    def reset(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        重置跟踪器
        
        Args:
            frame: 视频帧
            bbox: 新的目标边界框 (x, y, width, height)
            
        Returns:
            是否成功重置
        """
        self.create_new_tracker()
        return self.init(frame, bbox)

class MultiObjectTracker:
    """多目标跟踪器类"""
    
    def __init__(self, tracker_type: str = "csrt"):
        """
        初始化多目标跟踪器
        
        Args:
            tracker_type: 跟踪器类型
        """
        self.tracker_type = tracker_type
        self.trackers = []  # 跟踪器列表
        self.bboxes = []    # 边界框列表
        self.object_ids = []  # 目标ID列表
        self.next_id = 1    # 下一个目标ID
    
    def add_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        添加新的跟踪器
        
        Args:
            frame: 视频帧
            bbox: 目标边界框 (x, y, width, height)
            
        Returns:
            目标ID
        """
        tracker = ObjectTracker(self.tracker_type)
        success = tracker.init(frame, bbox)
        
        if success:
            object_id = self.next_id
            self.next_id += 1
            
            self.trackers.append(tracker)
            self.bboxes.append(bbox)
            self.object_ids.append(object_id)
            
            return object_id
        else:
            return -1
    
    def update(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        更新所有跟踪器
        
        Args:
            frame: 当前视频帧
            
        Returns:
            跟踪结果列表 [(object_id, bbox), ...]
        """
        # 更新每个跟踪器
        to_remove = []
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            
            if success:
                self.bboxes[i] = bbox
            else:
                # 如果跟踪失败，标记为删除
                to_remove.append(i)
        
        # 从后向前删除失败的跟踪器
        for i in sorted(to_remove, reverse=True):
            del self.trackers[i]
            del self.bboxes[i]
            del self.object_ids[i]
        
        # 返回跟踪结果
        return list(zip(self.object_ids, self.bboxes))
    
    def reset(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """
        重置所有跟踪器
        
        Args:
            frame: 视频帧
            bboxes: 新的目标边界框列表
            
        Returns:
            目标ID列表
        """
        # 清除所有跟踪器
        self.trackers = []
        self.bboxes = []
        self.object_ids = []
        
        # 添加新的跟踪器
        object_ids = []
        for bbox in bboxes:
            object_id = self.add_tracker(frame, bbox)
            object_ids.append(object_id)
        
        return object_ids

def draw_tracking_results(frame: np.ndarray, 
                         tracking_results: List[Tuple[int, Tuple[int, int, int, int]]],
                         show_id: bool = True) -> np.ndarray:
    """
    在视频帧上绘制跟踪结果
    
    Args:
        frame: 视频帧
        tracking_results: 跟踪结果列表 [(object_id, bbox), ...]
        show_id: 是否显示目标ID
        
    Returns:
        绘制结果图像
    """
    result = frame.copy()
    
    for object_id, (x, y, w, h) in tracking_results:
        # 为每个目标ID生成不同的颜色
        color = (
            (object_id * 50) % 255,
            (object_id * 100) % 255,
            (object_id * 150) % 255
        )
        
        # 绘制边界框
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # 如果需要，显示目标ID
        if show_id:
            cv2.putText(result, f"ID: {object_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def process_video(video_path: str, 
                 tracker_type: str = "csrt",
                 output_path: Optional[str] = None,
                 display: bool = False) -> None:
    """
    处理视频并进行目标跟踪
    
    Args:
        video_path: 视频文件路径
        tracker_type: 跟踪器类型
        output_path: 输出视频路径
        display: 是否显示处理结果
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, 共{total_frames}帧")
    
    # 创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"输出视频将保存到: {output_path}")
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    
    # 选择要跟踪的目标
    print("在第一帧上选择要跟踪的目标。按'Enter'确认选择，按'q'退出选择模式。")
    bboxes = []
    
    # 定义鼠标回调函数
    def select_roi(event, x, y, flags, param):
        nonlocal bboxes, frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始选择区域
            param['is_drawing'] = True
            param['start_x'] = x
            param['start_y'] = y
        
        elif event == cv2.EVENT_MOUSEMOVE and param['is_drawing']:
            # 绘制临时矩形
            img_copy = frame.copy()
            
            # 绘制已选择的边界框
            for bbox in bboxes:
                x1, y1, w, h = bbox
                cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            
            # 绘制当前正在选择的边界框
            x1, y1 = param['start_x'], param['start_y']
            cv2.rectangle(img_copy, (x1, y1), (x, y), (255, 0, 0), 2)
            
            cv2.imshow("选择跟踪目标", img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成选择区域
            param['is_drawing'] = False
            x1, y1 = param['start_x'], param['start_y']
            
            # 确保宽度和高度为正数
            w = abs(x - x1)
            h = abs(y - y1)
            x1 = min(x1, x)
            y1 = min(y1, y)
            
            # 添加边界框
            if w > 0 and h > 0:
                bboxes.append((x1, y1, w, h))
                
                # 显示已选择的边界框
                img_copy = frame.copy()
                for bbox in bboxes:
                    x1, y1, w, h = bbox
                    cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.imshow("选择跟踪目标", img_copy)
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("选择跟踪目标")
    callback_param = {'is_drawing': False, 'start_x': -1, 'start_y': -1}
    cv2.setMouseCallback("选择跟踪目标", select_roi, callback_param)
    
    # 显示第一帧并等待用户选择目标
    cv2.imshow("选择跟踪目标", frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        
        elif key == 13:  # Enter键
            break
    
    # 关闭选择窗口
    cv2.destroyWindow("选择跟踪目标")
    
    # 如果没有选择目标，退出
    if not bboxes:
        print("未选择任何目标，退出")
        return
    
    print(f"已选择 {len(bboxes)} 个目标进行跟踪")
    
    # 创建多目标跟踪器
    tracker = MultiObjectTracker(tracker_type)
    object_ids = tracker.reset(frame, bboxes)
    
    print(f"已初始化跟踪器，目标ID: {object_ids}")
    
    # 处理视频的其余部分
    frame_count = 1
    start_time = time.time()
    
    # 如果需要显示结果，创建窗口
    if display:
        cv2.namedWindow("跟踪结果")
    
    while True:
        # 读取下一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 更新跟踪器
        tracking_results = tracker.update(frame)
        
        # 绘制跟踪结果
        result_frame = draw_tracking_results(frame, tracking_results)
        
        # 添加帧计数器
        cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示结果
        if display:
            cv2.imshow("跟踪结果", result_frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 写入输出视频
        if writer:
            writer.write(result_frame)
        
        # 打印进度
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_real = frame_count / elapsed_time
            print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%), "
                 f"实际fps: {fps_real:.1f}")
    
    # 清理资源
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # 打印处理统计信息
    elapsed_time = time.time() - start_time
    print(f"处理完成，共处理 {frame_count} 帧，用时 {elapsed_time:.2f} 秒，"
         f"平均 {frame_count/elapsed_time:.2f} fps")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="视频目标跟踪程序")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--tracker", choices=list(ObjectTracker.TRACKER_TYPES.keys()),
                       default="csrt", help="跟踪器类型")
    parser.add_argument("--display", action="store_true", help="是否显示处理结果")
    args = parser.parse_args()
    
    # 处理视频
    process_video(args.video, args.tracker, args.output, args.display)

if __name__ == "__main__":
    main() 