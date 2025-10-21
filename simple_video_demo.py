#!/usr/bin/env python3
"""
简化版视频补帧Demo
快速测试IFRNet视频补帧功能

python simple_video_demo.py in.mp4 out.mp4
"""

import os
import cv2
import numpy as np
import torch
import imageio
from models.IFRNet import Model

# Create alias for the model class
IFRNet = Model


def simple_video_interpolation(input_video, output_video=None, model_path='checkpoints/IFRNet/IFRNet_Vimeo90K.pth'):
    """
    简单的视频补帧函数
    输入: 原视频路径
    输出: 补帧后的视频 (帧率翻倍，时长不变)
    """
    
    # 处理输出文件名
    if output_video is None:
        base_name = os.path.splitext(input_video)[0]
        output_video = f"{base_name}_interpolated.avi"  # 使用AVI格式
    elif not output_video.endswith('.avi'):
        # 强制使用AVI格式以避免编码问题
        base_name = os.path.splitext(output_video)[0]
        output_video = f"{base_name}.avi"
    
    # 加载模型 (参考demo_2x.py的正确方式)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("加载IFRNet模型...")
    if device.type == 'cuda':
        model = Model().cuda().eval()
    else:
        model = Model().eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型加载完成!")
    
    # 使用imageio读取视频
    print("读取输入视频...")
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']
    frames = []
    
    # 读取所有帧
    for frame in reader:
        frames.append(frame)
    reader.close()
    
    total_frames = len(frames)
    if total_frames < 2:
        print("❌ 视频帧数不足，需要至少2帧")
        return
    
    height, width = frames[0].shape[:2]
    
    # 确保尺寸能被16整除（为了视频编码兼容性）
    new_width = (width + 15) // 16 * 16
    new_height = (height + 15) // 16 * 16
    
    # 如果需要调整尺寸，重新调整所有帧
    if new_width != width or new_height != height:
        print(f"调整视频尺寸从 {width}x{height} 到 {new_width}x{new_height} (确保能被16整除)")
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            resized_frames.append(resized_frame)
        frames = resized_frames
        width, height = new_width, new_height
    
    print(f"输入视频设置: {width}x{height}, {fps:.2f}FPS, {total_frames}帧")
    
    # 设置输出视频 (帧率翻倍，帧数也翻倍，时长保持不变)
    output_fps = fps * 2  # 帧率翻倍
    
    print(f"输出视频设置: {width}x{height}, {output_fps:.2f}FPS")
    
    # 预处理函数 (参考demo_2x.py的实现)
    def preprocess(frame):
        # imageio读取的帧已经是RGB格式，不需要转换
        frame_tensor = (torch.tensor(frame.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)
        if device.type == 'cuda':
            frame_tensor = frame_tensor.cuda()
        return frame_tensor
    
    # 准备输出帧列表
    output_frames = []
    
    # 添加第一帧
    output_frames.append(frames[0])
    print(f"添加第1帧 (原始帧)")
    
    print("开始处理视频...")
    
    # 处理每对相邻帧
    for i in range(len(frames) - 1):
        prev_frame = frames[i]
        curr_frame = frames[i + 1]
        
        # 转换为张量
        img0 = preprocess(prev_frame)
        img1 = preprocess(curr_frame)
        
        # 生成中间帧 (t=0.5, 参考demo_2x.py)
        embt = torch.tensor(1/2).view(1, 1, 1, 1).float()
        if device.type == 'cuda':
            embt = embt.cuda()
        
        with torch.no_grad():
            imgt_pred = model.inference(img0, img1, embt)
        
        # 后处理中间帧 (imageio使用RGB格式，不需要转换)
        interp_frame = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        
        # 添加插值帧和当前帧
        output_frames.append(interp_frame)
        output_frames.append(curr_frame)
        
        print(f"处理第{i+1}-{i+2}帧，生成插值帧")
    
    print(f"已处理 {len(frames)}帧，生成 {len(output_frames)}帧")
    
    # 使用imageio写入视频
    print("写入输出视频...")
    imageio.mimsave(output_video, output_frames, fps=output_fps)
    
    print(f"✅ 视频补帧完成!")
    print(f"输出文件: {output_video}")
    print(f"输出视频: {output_fps:.2f}FPS (帧率翻倍，时长不变，更流畅)")
    print(f"总共生成: {len(output_frames)}帧 (原始{total_frames}帧 → 补帧后{len(output_frames)}帧)")


if __name__ == "__main__":
    import sys, time
    
    if len(sys.argv) < 2:
        print("用法: python simple_video_demo.py <输入视频> [输出视频]")
        print("示例: python simple_video_demo.py input.mp4 output_2x.mp4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output_interpolated.mp4"
    
    if not os.path.exists(input_video):
        print(f"错误: 输入文件不存在: {input_video}")
        sys.exit(1)
    
    try:
        # 开始时间
        start_time = time.time()
        simple_video_interpolation(input_video, output_video)
        # 结束时间
        end_time = time.time()
        # 计算耗时
        elapsed_time = end_time - start_time
        print(f"耗时: {elapsed_time:.2f}秒")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)