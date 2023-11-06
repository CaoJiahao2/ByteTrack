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
### tutorial
### yolox