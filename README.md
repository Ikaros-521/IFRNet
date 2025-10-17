# IFRNet: 用于高效帧插值的中间特征细化网络
[IFRNet](https://arxiv.org/abs/2205.14620) (CVPR 2022) 的官方 PyTorch 实现。

作者：[Lingtong Kong](https://scholar.google.com.hk/citations?user=KKzKc_8AAAAJ&hl=zh-CN), [Boyuan Jiang](https://byjiang.com/), Donghao Luo, Wenqing Chu, Xiaoming Huang, [Ying Tai](https://tyshiwo.github.io/), Chengjie Wang, [Jie Yang](http://www.pami.sjtu.edu.cn/jieyang)

## 亮点
现有的基于光流的帧插值方法几乎都首先估计或建模中间光流，然后使用光流扭曲的上下文特征来合成目标帧。然而，它们忽略了中间光流和中间上下文特征的相互促进作用。此外，它们的级联架构会大幅增加推理延迟和模型参数，阻碍了它们在许多移动和实时应用中的使用。我们首次将上述分离的光流估计和上下文特征细化合并到一个基于编码器-解码器的 IFRNet 中，以实现紧凑性和快速推理，其中这两个关键元素可以相互受益。此外，我们新提出了面向任务的光流蒸馏损失和特征空间几何一致性损失，分别用于促进 IFRNet 的中间运动估计和中间特征重建。基准测试结果表明，我们的 IFRNet 不仅实现了最先进的 VFI 精度，而且还具有快速推理速度和轻量级模型尺寸。

![](./figures/vimeo90k.png)

## YouTube 演示
[[4K60p] うたわれるもの 偽りの仮面 OP フレーム補間+超解像 (IFRnetとReal-CUGAN)](https://www.youtube.com/watch?v=tV2imgGS-5Q)

[[4K60p] 天神乱漫 -LUCKY or UNLUCKY!?- OP (IFRnetとReal-CUGAN)](https://www.youtube.com/watch?v=NtpJqDZaM-4)

[RIFE IFRnet 比較](https://www.youtube.com/watch?v=lHqnOQgpZHQ)

[IFRNet 帧插值](https://www.youtube.com/watch?v=ygSdCCZCsZU)

## 准备
1. PyTorch >= 1.3.0 (我们已验证此仓库支持 Python 3.6/3.7, PyTorch 1.3.0/1.9.1)。
2. 下载训练和测试数据集：[Vimeo90K](http://toflow.csail.mit.edu/), [UCF101](https://liuziwei7.github.io/projects/VoxelFlow), [SNU-FILM](https://myungsub.github.io/CAIN/), [Middlebury](https://vision.middlebury.edu/flow/data/), [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) 和 [Adobe240](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)。
3. 在您的机器上设置正确的数据集路径。

## 下载预训练模型并运行演示
从左到右的图片分别是叠加的输入帧、2倍和8倍视频插值结果。
<p float="left">
  <img src=./figures/img_overlaid.png width=270 />
  <img src=./figures/out_2x.gif width=270 />
  <img src=./figures/out_8x.gif width=270 /> 
</p>

1. 从此[链接](https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0)下载我们的预训练模型，然后将文件 `checkpoints` 放入根目录。

2. 运行以下脚本生成2倍和8倍帧插值演示
<pre><code>$ python demo_2x.py</code>
<code>$ python demo_8x.py</code></pre>


## 在 Vimeo90K 三元组数据集上进行2倍帧插值训练
1. 首先，运行此脚本生成光流伪标签
<pre><code>$ python generate_flow.py</code></pre>

2. 然后，通过执行以下命令之一并选择模型开始训练
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_L' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_S' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>

## 2倍帧插值基准测试
要测试运行时间和模型参数，您可以运行
<pre><code>$ python benchmarks/speed_parameters.py</code></pre>

要在 Vimeo90K、UCF101 和 SNU-FILM 数据集上测试帧插值精度，您可以运行
<pre><code>$ python benchmarks/Vimeo90K.py</code>
<code>$ python benchmarks/UCF101.py</code>
<code>$ python benchmarks/SNU_FILM.py</code></pre>

## 2倍帧插值定量比较
提出的 IFRNet 以更少的推理时间和计算复杂度实现了最先进的帧插值精度。我们期望提出的基于单编码器-解码器联合细化的 IFRNet 成为许多帧率上转换、视频压缩和中间视图合成系统的有用组件。时间和 FLOPs 在 1280 x 720 分辨率下测量。

![](./figures/benchmarks.png)


## 2倍帧插值定性比较
在 SNU-FILM 数据集上使用2个输入帧的方法进行2倍插值的视频比较。

![](./figures/fig2_1.gif)

![](./figures/fig2_2.gif)


## Middlebury 基准测试
[Middlebury](https://vision.middlebury.edu/flow/eval/results/results-i1.php) 在线基准测试结果。

![](./figures/middlebury.png)

Middlebury Other 数据集的结果。

![](./figures/middlebury_other.png)


## 在 GoPro 数据集上进行8倍帧插值训练
1. 通过执行以下命令之一并选择模型开始训练
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_gopro.py --world_size 4 --model_name 'IFRNet' --epochs 600 --batch_size 2 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_gopro.py --world_size 4 --model_name 'IFRNet_L' --epochs 600 --batch_size 2 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_gopro.py --world_size 4 --model_name 'IFRNet_S' --epochs 600 --batch_size 2 --lr_start 1e-4 --lr_end 1e-5</code></pre>

由于8倍插值设置中的帧间运动相对较小，此处省略了面向任务的光流蒸馏损失。由于 GoPro 训练集是一个相对较小的数据集，我们建议使用您的特定数据集来训练慢动作生成以获得更好的结果。

## 8倍帧插值定量比较

<img src=./figures/8x_interpolation.png width=480 />

## GoPro 和 Adobe240 数据集上8倍帧插值的定性结果
每个视频有9帧，其中第一帧和最后一帧是输入，中间7帧由 IFRNet 预测。

<p float="left">
  <img src=./figures/fig1_1.gif width=270 />
  <img src=./figures/fig1_2.gif width=270 />
  <img src=./figures/fig1_3.gif width=270 /> 
</p>

## IFRNet 的 ncnn 实现

[ifrnet-ncnn-vulkan](https://github.com/nihui/ifrnet-ncnn-vulkan) 使用 [ncnn 项目](https://github.com/Tencent/ncnn) 作为通用神经网络推理框架。此包包含所有必需的二进制文件和模型。它是可移植的，因此不需要 CUDA 或 PyTorch 运行时环境。

## 引用
在您的工作中使用本软件或论文的任何部分时，请引用以下论文：
<pre><code>@InProceedings{Kong_2022_CVPR, 
  author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie}, 
  title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation}, 
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year = {2022}
}</code></pre>
