import cv2
import torch
import numpy as np
import time
import threading
import queue
import argparse
import os
from models.IFRNet import Model
from utils import read # Assuming utils.py has a 'read' function that can handle image loading

# 尝试导入ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: 未安装onnxruntime，将使用PyTorch推理")
    print("要使用ONNX加速，请运行: pip install onnxruntime-gpu")

class ONNXModel:
    """ONNX模型推理包装器"""
    def __init__(self, onnx_path):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime未安装")
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
        
        # 设置执行提供者（优先使用GPU）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"ONNX模型加载成功，使用提供者: {self.session.get_providers()}")
        
        # 获取输入输出信息
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def inference(self, img0, img1, embt):
        """
        ONNX模型推理
        Args:
            img0: torch.Tensor, 第一帧图像
            img1: torch.Tensor, 第二帧图像  
            embt: torch.Tensor, 时间嵌入
        Returns:
            torch.Tensor: 插值结果
        """
        # 转换为numpy数组
        img0_np = img0.cpu().numpy()
        img1_np = img1.cpu().numpy()
        embt_np = embt.cpu().numpy()
        
        # 准备输入
        inputs = {
            'img0': img0_np,
            'img1': img1_np, 
            'embt': embt_np
        }
        
        # 运行推理
        outputs = self.session.run(None, inputs)
        
        # 转换回torch tensor
        result = torch.from_numpy(outputs[0]).cuda()
        return result

def preprocess_frame(frame, target_size=None):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Store original size for later restoration
    original_h, original_w = frame_rgb.shape[:2]
    
    # If target_size is specified, resize to that size
    # Otherwise, keep original size but ensure dimensions are divisible by 32 for the model
    if target_size is not None:
        frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
    else:
        # Ensure dimensions are divisible by 32 (common requirement for deep learning models)
        h, w = original_h, original_w
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        if h != new_h or w != new_w:
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to PyTorch tensor
    img_tensor = (torch.tensor(frame_rgb.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
    return img_tensor, (original_h, original_w)

def postprocess_frame(img_tensor, original_size=None):
    # Convert PyTorch tensor back to OpenCV BGR image
    img_np = (img_tensor[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Resize back to original size if specified
    if original_size is not None:
        original_h, original_w = original_size
        img_bgr = cv2.resize(img_bgr, (original_w, original_h), interpolation=cv2.INTER_AREA)
    
    return img_bgr

def camera_thread(cap, frame_queue, fps_stats):
    """独立的摄像头读取线程，计算真实摄像头帧率"""
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
    
    while fps_stats['running']:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time()
        
        # 计算真实摄像头帧率
        elapsed = current_time - last_fps_update
        if elapsed >= 1.0:  # 每秒更新一次
            fps_stats['camera_fps'] = frame_count / elapsed
            frame_count = 0
            last_fps_update = current_time
        
        # 将帧放入队列，如果队列满了就丢弃旧帧
        try:
            frame_queue.put_nowait((frame, current_time))
        except queue.Full:
            try:
                frame_queue.get_nowait()  # 移除旧帧
                frame_queue.put_nowait((frame, current_time))
            except queue.Empty:
                pass

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='IFRNet实时帧插值演示')
    parser.add_argument('--use-onnx', action='store_true', help='使用ONNX模型进行推理加速')
    parser.add_argument('--onnx-path', default='./checkpoints/IFRNet/IFRNet_Vimeo90K.onnx', 
                       help='ONNX模型路径')
    parser.add_argument('--pytorch-path', default='./checkpoints/IFRNet/IFRNet_Vimeo90K.pth',
                       help='PyTorch模型路径')
    args = parser.parse_args()
    
    # 加载模型 (暂时禁用ONNX，因为存在形状兼容性问题)
    model = None
    model_type = "PyTorch"
    
    # 暂时注释掉ONNX部分，直到解决形状问题
    # if args.use_onnx and ONNX_AVAILABLE:
    #     try:
    #         # 检查ONNX模型是否存在
    #         if not os.path.exists(args.onnx_path):
    #             print(f"ONNX模型不存在: {args.onnx_path}")
    #             print("请先运行 python convert_to_onnx.py 来转换模型")
    #             return
    #         
    #         model = ONNXModel(args.onnx_path)
    #         model_type = "ONNX"
    #         print("使用ONNX模型进行推理")
    #     except Exception as e:
    #         print(f"ONNX模型加载失败: {e}")
    #         print("回退到PyTorch模型")
    #         args.use_onnx = False
    
    # 使用优化的PyTorch模型
    try:
        model = Model().cuda().eval()
        model.load_state_dict(torch.load(args.pytorch_path))
        
        # 启用PyTorch优化
        torch.backends.cudnn.benchmark = True  # 优化CUDNN性能
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提高性能
        
        print("使用优化的PyTorch模型进行推理")
        if args.use_onnx:
            print("注意: ONNX模型暂时不可用，正在使用PyTorch模型")
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please download the pre-trained models as instructed in README.md.")
        print("Download link: https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0")
        return
    except Exception as e:
        print(f"PyTorch模型加载失败: {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution (optional, but good for consistency)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit.")

    # 创建帧队列和FPS统计字典
    frame_queue = queue.Queue(maxsize=5)  # 限制队列大小避免内存问题
    fps_stats = {
        'running': True,
        'camera_fps': 0
    }
    
    # 启动摄像头读取线程
    camera_t = threading.Thread(target=camera_thread, args=(cap, frame_queue, fps_stats))
    camera_t.daemon = True
    camera_t.start()
    
    # Variables for processing FPS calculation
    processing_fps_start = time.time()
    processing_frame_count = 0
    output_frame_count = 0
    
    # Previous frame for interpolation
    prev_frame_tensor = None
    prev_original_size = None

    try:
        while fps_stats['running']:
            try:
                # 从队列获取最新帧
                current_frame_np, frame_timestamp = frame_queue.get(timeout=1.0)
                processing_frame_count += 1
                
                current_frame_tensor, original_size = preprocess_frame(current_frame_np)
                
                # Calculate processing FPS
                current_time = time.time()
                elapsed_time = current_time - processing_fps_start
                processing_fps = processing_frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # If we have a previous frame, generate interpolated frame
                if prev_frame_tensor is not None:
                    # Perform inference for interpolation
                    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()
                    with torch.no_grad():
                        interpolated_frame_tensor = model.inference(prev_frame_tensor, current_frame_tensor, embt)

                    # Post-process interpolated frame
                    display_interpolated = postprocess_frame(interpolated_frame_tensor, original_size)
                    
                    # Calculate output FPS (should be ~2x processing)
                    output_frame_count += 2  # Current frame + interpolated frame
                    output_fps = output_frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Add text overlays
                    camera_fps_text = f"Camera: {int(fps_stats['camera_fps'])} FPS"
                    processing_fps_text = f"Processing: {int(processing_fps)} FPS"
                    output_fps_text = f"Output: {int(output_fps)} FPS"
                    model_text = f"Model: {model_type}"
                    
                    cv2.putText(display_interpolated, camera_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_interpolated, processing_fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_interpolated, output_fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_interpolated, model_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_interpolated, "IFRNet 2x Interpolated", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display interpolated frame
                    cv2.imshow('IFRNet 2x Interpolation', display_interpolated)
                else:
                    # For the first frame, just display the original
                    display_current = postprocess_frame(current_frame_tensor, original_size)
                    cv2.putText(display_current, "Initializing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('IFRNet 2x Interpolation', display_current)
                    output_frame_count += 1
                
                # Update previous frame for next iteration
                prev_frame_tensor = current_frame_tensor
                prev_original_size = original_size
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except queue.Empty:
                # 如果队列为空，继续等待
                continue
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # 停止摄像头线程
        fps_stats['running'] = False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()