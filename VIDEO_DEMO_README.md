# 视频补帧Demo使用说明

本项目提供了两个视频补帧demo脚本，基于IFRNet模型实现视频帧率提升。

## 文件说明

### `simple_video_demo.py` - 简化版Demo
轻量级的视频补帧工具，专注于核心功能。

**特性:**
- 固定2倍帧率插值
- 简单易用
- 代码结构清晰，便于学习

**使用方法:**
```bash
# 基本用法
python simple_video_demo.py input.mp4

# 指定输出文件
python simple_video_demo.py input.mp4 output.mp4
```

## 环境要求

### 必需依赖
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install tqdm
```

### 模型文件
确保以下模型文件存在:
- `./checkpoints/IFRNet/IFRNet_Vimeo90K.pth`

如果模型文件不存在，请从项目官方仓库下载。

## 支持的视频格式

**输入格式:**
- MP4, AVI, MOV, MKV等常见视频格式
- 建议使用MP4格式以获得最佳兼容性

**输出格式:**
- MP4 (使用mp4v编码器)

## 性能说明

### GPU vs CPU
- **GPU (推荐)**: 处理速度快，适合高分辨率视频
- **CPU**: 处理速度较慢，但兼容性更好

### 内存使用
- 视频分辨率越高，内存使用越多
- 建议处理4K视频时确保有足够的GPU内存

### 处理时间估算
以1080p视频为例:
- GPU (RTX 3080): 约1-2分钟/分钟视频
- CPU (i7-10700K): 约10-20分钟/分钟视频

## 示例用法

### 示例1: 基本2倍帧率提升
```bash
python simple_video_demo.py sample.mp4 sample_2x.mp4
```


## 两种补帧模式说明

### 🎯 标准补帧模式 (默认，推荐)
- **效果**: 帧率翻倍，视频时长不变，播放更加流畅
- **原理**: 在原有帧之间插入新帧，同时提高播放帧率
- **适用**: 大多数情况，这是正确的视频补帧效果
- **示例**: 30fps 10秒视频 → 60fps 10秒视频 (帧数翻倍，时长不变)

