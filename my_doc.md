# 代码阅读
## assets ()
主要包含一些gif文件
## datasets
### data_path
关于训练图片的记录
## deploy
1. DeepStream
nvidia的视频流处理框架，可以用来部署模型,快速开发和部署视觉AI应用与服务
2. TensorRT
nvidia的深度学习推理框架，可以用来部署模型
3. ncnn
腾讯的深度学习推理框架，可以用来部署模型，高效易用的深度学习推理框架，支持各自神经网络模型
4. ONNXRuntime
微软的深度学习推理框架，可以用来部署模型，高性能的推理引擎，支持各种硬件平台
## exps，该文件之间从yolox官方文件复制过来
### default
nano.py:更轻量化，以面向CPU和移动端的低算力设备的yolo检测器，nona：纳米
yolox_tiny.py
yolox_s.py
yolox_m.py
yolox_l.py
yolox_x.py
以上文件主要做数据处理，均继承继承了YOLOX的类
### example
#### mot
#### uot
### tools
将各种数据集转换为coco格式
convert_cityperson_to_coco.py
视频转化
convert_video.py

### tutorial
### videos
包含了一个天安门的视频
### yolox
yolox包
### kalman_filter.py
卡尔曼滤波是一种递归的状态估计方法，用于处理具有线性动态系统和高斯噪声的问题。其基本思想是通过融合系统动态模型和实际观测值，逐步更新对系统状态的估计。
以下是卡尔曼滤波的基本过程和思想：
初始化： 首先，初始化状态的均值向量和协方差矩阵。这通常由问题的先验信息提供，例如初始位置、速度和不确定性的估计。
预测： 利用系统的动态模型，通过运动方程对当前状态进行预测，得到下一个时刻的状态的预测均值向量和协方差矩阵。预测步骤中考虑了系统的运动和相应的不确定性。
校正（更新）： 利用实际观测值，通过观测方程对预测的状态进行校正，得到对当前状态的更新估计。卡尔曼增益用于权衡预测和观测，更加相信具有更小不确定性的那一方。
迭代： 重复预测和校正步骤，逐步更新状态的估计值，同时降低估计的不确定性。