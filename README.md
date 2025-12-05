# java_pytorch_onnx_tensorrt_yolov8_vue 
  视频算法中台系统

#### 介绍
    

    1.基于SpringBoot+Docker+Cuda+Cudnn+Pythorch+Onnx+Tensorrt+Yolov8+ffmpeg+zlmediakit 的AI算法中台系统，本系统主要实现JAVA调用Python脚本的方式，实现在GPU(Nvidia Tesla T4)上进行yolov8的加速推理运算。

    2.项目可实现人、车、火灾烟雾、河道漂浮物、道路裂痕等视频的实时识别，并将识别结果通过ffmpeg推流到zlmediakit流媒体服务器，使得在web页面上可以同时查看原始视频和实时计算视频。

    3.项目每隔1分钟将推理结果信息和对应的图片推送到文件服务器minio和消息队列rocketmq,便于开发者获取到推理结果进行业务开发。
    
    4.项目同时支持基于ONNX的推理运算和基于Tensorrt的加速推理运算2种方式,只需在调用时传递不同参数即可。

    5.项目同时支持Windows环境，代码自动判断运行的环境并执行对应的.bat或.sh脚本文件以启动AI模型推理。

    6.项目的核心是yolov8神经网络模型的推理运算，推理运算涉及操作cpu内存、gpu内存、gpu并行计算等环节，这些环节可通过python或c++来实现，作者对2种方式均进行了实现

  
    

#### 软件架构

1.系统架构
    本AI算法中台系统结合了SpringBoot开发Web项目的优势、Python训练调试AI模型的优势、C++在与底层GPU硬件交互和推理运算方面的性能优势、Shell脚本便于调用Linux服务器资源的优势，将不同语言综合一起协调工作，并将项目运行在Docker容器中以便于运维和管理，让JAVA开发者能够方便训练、部署、使用AI模型。

![输入图片说明](img/AI.png)


2.技术栈

开发环境：IDEA、JDK1.8、Maven、Gitlab、Pycharm、Anaconda3

软件架构：Nginx + SpringBoot + Vue + Shell + Python

前端框架：Vue + Nodejs

Web框架：SpringBoot

Orm框架：MyBatis3

数据库：Mysql、Redis、MongoDB

消息队列：RocketMq

文件服务: Minio

代码生成: FreeMarker自动生成后端Java代码和前端Vue代码

CPU设备: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz

GPU设备: Nvidia Tesla T4 16G

AI模型：Yolov8n/Yolov8m

训练环境：Pytorch2.3.0 + Onnxruntime-gpu1.16.1 + Ultralytics8.2.48

推理环境: Cuda11.8 + Cudnn8.6.0 + Tensorrt8.5.1.7 + Onnxruntime-gpu1.16.1

推理加速: Tensorrt8.5.1.7

视频操作：Opencv4.7.0

视频推流: FFmpeg4.2.7

流媒体服务：Zlmediakit

部署环境：Docker + Ubuntu20.04




#### 目录说明

data/algorithm_model : 算法模型库，提供了作者训练并部署的5个常用算法模型,包括.pt .onnx .engine 三种格式

data/doc : 算法中台接口文档

data/front_code : 项目前端Vue代码

data/linux : linux下的onnx和tensorrt推理运行代码

data/windows : windows下的onnx和tensorrt推理运行代码

data/sql : 项目数据库

src : 项目后端springboot代码

3.运行前端项目 

  (1).本地VSCODE中运行：

      1).安装依赖 npm install
      
      2).修改服务器的访问地址

      3).启动项目 npm run dev
     
      4).访问地址 http://ip:8080/login  用户名/密码: admin / 123456

使用说明

1.算法中台的使用

  (1).前后端代码运行起来后，打开页面 http://ip:8080/login  用户名/密码: admin / 123456

  (2).点击"算法模型"即可上传修改删除训练好的AI模型

  (3).点击"计算任务"即可查看需要计算的任务，点击启用即可开始进行AI模型的推理运算

  (4).点击"告警中心"即可查看模型的告警记录,比如识别到河道有垃圾漂浮物即产生一条告警信息

  (5).点击"推送日志"即可查看本平台向其他平台推送的HTTP告警日志记录

  (6).点击"客户管理"即可查看需要给哪些客户平台推送告警信息

  (7).打开文件data/doc_sql/算法中台接口文档(v2.0).docx 即可通过接口实现与算法中台的对接

2.模型的训练
 
  (1).Yolov8 预训练模型下载 https://github.com/ultralytics/ultralytics

  (2).自己准备数据或在kaggle、 CV Datasets on the web、 阿里天池数据集等深度学习数据集下载网站下载数据

  (3).将数据集分为训练集、验证集、测试集，使用Pytorch加载Yolov8预训练模型并训练自己的数据

  (4).查看训练结果，主要关注精确度Precision、召回率Recall、P-R曲线、F1 Score、mAP50、mAP50-95等重要参数

  (5).根据模型训练结果判断是否需要进行再次的调优训练，例如修改Epoch、调整训练集、测试集的比例等等

  (6).训练完成后得到当前最佳的模型best.pt

  (7).使用Python脚本调用best.pt运行测试集数据进行模型的验证

3.模型的部署

  (1).将训练好的best.pt拷贝到容器内部的 /data/app/yolo/tensorrt_infer/YOLOv8-TensorRT-main目录下

  (2).进入到/data/app/yolo/tensorrt_infer/YOLOv8-TensorRT-main目录,通过如下命令将 best.pt 转换为 best.onnx模型格式
      python3 export-det.py --weights best.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640
  
  (3).接着执行如下命令将best.onnx转为tensorrt的引擎文件格式best.engine，以便实现在nvidia gpu上进行推理加速
      /gsis_ai/nvidia/TensorRT-8.5.1.7/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

  (4).在算法中台管理界面点击"算法模型-添加"，填写模型基本信息，视频地址可以是rtsp/rtmp/flv/hls/http/ws等任意格式的视频地址或离线视频文件.mp4，注意模型键(modelKey)为刚才生成的引擎文件的文件名best,脚本键(shellKey)为yolov8

  (5).填写好模型信息后，点击文件上传，将转换生成好的best.engine上传到minio中，后期运行该模型时系统会自动下载该模型到对应的工作目录(比如 /data/app/yolo/tensorrt_infer/)

  (6).在计算任务中点击 "新增"选择模型，添加对应的视频地址，然后保存

  (7).在任务列表点击"启用"即可加载模型并进行推理

  (8).启用成功后点击最右边视频图标，即可查看原始视频画面和实时推理画面



效果展示
![输入图片说明](img/login.png)


![输入图片说明](img/first.png)








![输入图片说明](img/car2.png)


![输入图片说明](img/fire.png)


![输入图片说明](img/safehat.png)









相关开源

常康:Java-Onnx-Yolo

https://toscode.mulanos.cn/agricultureiot/yolo-onnx-java



xgc:JavaVision


https://gitee.com/giteeClass/java-vision




