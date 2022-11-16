# SNPETask

[![](https://img.shields.io/badge/Auther-@RicardoLu-red.svg)](https://github.com/gesanqiu)![](https://img.shields.io/badge/Version-2.1.0-blue.svg)[![](https://img.shields.io/github/stars/gesanqiu/SNPE_Tutorial.svg?style=social&label=Stars)](https://github.com/gesanqiu/SNPE_Tutorial)

欢迎各位star/fork本repo并进行学习交流。

【注】：提Support Request/New Requirement前请先star。

## Overview

![1656919889506](images/1656919889506.png)

上图源自Snapdragon Neural Processing Engine SDK Reference Guide，它展示了一个Deep Learning Neural Network在SNPE环境下的Workflow。本repo主要关注Workflow中的第二个阶段，当你拥有一个预训练模型（以`yolov5s.onnx`为例），如何从Model Convert到SNPE Enabled App，如官方教程所言，主要有以下四个步骤：

- [Model Convert](./doc/ModelConvert.md)：把`.tf/.tflite/.onnx/caffe/caffe2/.pt`等网络预训练模型转换为一个能够被SNPE加载的DLC文件。
- [Benchmark](./doc/Benchmark.md)：量化模型，以便能在Hexagon DSP上运行（可选项）。 
- [SNPETask](./doc/SNPETask.md)：使用SNPE runtime加载模型并完成推理。
- [YOLOv5s](./doc/YOLOv5s.md)：Pre-process（处理输入）和Post-process（处理输出）。

## File Tree

```shell
root@tc-eb5:/home/tc-eb5/local/SNPETask# tree
.
├── CMakeLists.txt
├── LICENSE
├── README.md
├── build.sh                  # Build Script.
├── doc                       # Tutorials.
│   ├── Benchmark.md
│   ├── FAQ.md
│   ├── ModelConvert.md
│   ├── SNPETask.md
│   └── YOLOv5s.md
├── lib                       # ELF of YOLOv5s SDK.
│   └── libYOLOv5s.so
├── model
│   ├── yolov5s.dlc
│   └── yolov5s_labels.txt
├── snpetask                  # SNPE runtime SDK source codes.
│   ├── SNPETask.cpp
│   └── SNPETask.h
├── test                      # Test program.
│   ├── test_image            # Image input.
│   │   ├── CMakeLists.txt
│   │   ├── config.json
│   │   ├── main.cpp
│   │   ├── orange.jpeg
│   │   └── people.jpg
│   └── test_video            # Stream input.
│       ├── CMakeLists.txt
│       ├── Flowchart.jpg
│       ├── SafeQueue.h
│       ├── VideoAnalyzer.cpp
│       ├── VideoAnalyzer.h
│       ├── VideoPipeline.cpp
│       ├── VideoPipeline.h
│       ├── config.json
│       ├── main.cpp
│       └── yolov5s_thresholds.txt    # Class base threshold.
├── utility
│   ├── Logger.h              # Singleton logger based on spdlog library.
│   └── utils.h
└── yolov5s                   # YOLOv5s Inference SDK source codes.
    ├── CMakeLists.txt
    ├── inc
    │   ├── YOLOv5s.h
    │   └── YOLOv5sImpl.h
    ├── src
    │   ├── YOLOv5s.cpp
    │   └── YOLOv5sImpl.cpp
    └── yolov5s.json
```

## Prerequisites

- 开发平台：Qualcomm® QRB5165 (Linux-Ubuntu 18.04)
- 图形界面：Weston(Wayland)
- 开发框架：Gstreamer-1.14.5，OpenCV-4.5.5
- 算法引擎：snpe-1.61.0.3358(不支持1.62及以上版本的SNPE)
- 算法示例模型：YOLOv5s.onnx
- 第三方库：gflags，json-glib-1.0，glib-2.0，spdlog-1.10.0，jsoncpp-1.7.4，mosquitto，mosquitto-clients
- 构建工具：CMake-3.10.2
- 打包工具：checkinstall-1.6.2

```shell
# SNPE Runtime Library需要用户从官方链接下载并放到开发板指定目录
# CMake/OpenCV/spdlog需要用户编译安装
# yolov5s.onnx预训练模型可以从官方repo通过export.py获取，详细流程请阅读：./doc/ModelConvert.md
# 其余依赖可以使用apt-get install安装
# sudo apt-get install libjson-glib-dev libgflags-dev libjsoncpp-dev libmosquitto-dev mosquitto mosquitto-clients
```

## Build & Compile

项目使用`CMakeLists.txt`进行构建和包管理，并提供了`build.sh`脚本用于构建整个项目，运行成功将输出如下log：

![1658202300948](images/1658202300948.png)

假如编译失败则会输出`[failed]`，相应的build deb package也会失败。

【注】：`build.sh`主要功能在于打包，不处理编译依赖。例如`OpenCV`需要用户修改`CMakeLists.txt`中中的`OpenCV_DIR`。

【注】：`build.sh`的打包功能依赖于`checkinstall`，需要确保make install能够成功运行，打包才能成功。

`build`目录下将包含如下信息，相关.so文件会被重定向到`${CMAKE_SOURCE_DIR}/lib`目录：

![1657616660419](images/1657616660419.png)

其中`test-yolov5`为测试程序，`alg-yolov5s-01-linux_2.0-rel_arm64.deb`为`checkinstall`追踪make install生成的deb安装包。

`alg-yolov5s-01-linux_2.0-rel_arm64.deb`可以直接使用`dpkg - i`命令安装，内含YOLOV5S SDK和AIRunner ALG两部分内容，均被安装至`/opt/thundersoft/`目录下，详细安装路径可以参考`CMakeLists.txt`中的install语句。

## Run Test Program

### test_image

测试程序`test-yolov5`使用gflags库进行命令行参数解析，可配置的参数及默认值如下：

```shell
Flags from /home/tc-eb5/local/SNPE_Tutorial/test/main.cpp:
    -confidence (Confidence Threshold.) type: double default: 0.5
    -config_path (Model config file path.) type: string
      default: "./yolov5s.json"
    -input (Input image file for this test program.) type: string
      default: "./image.jpg"
    -labels (Labels file for the yolov5s model.) type: string
      default: "./labels.txt"
    -nms (NMS Threshold.) type: double default: 0.5
```

测试程序将根据用户运行参数来初始化YOLOV5S Inference SDK对象并对输入图像进行推理，以下是在`build`目录下的一个运行样例：

```shell
./test/test_image test-image --input ../test/people.jpg --labels ../model/yolov5s_labels.txt --config_path ../yolov5s/yolov5s.json
```

上述命令以`test`目录下的`people.jpg`为待检测图片，使用`model`目录下的`yolov5s_labels.txt`作为模型类别输入，`yolov5s`目录下的`yolov5s.json`作为模型配置文件。

`yolov5s.json`为模型的一些基础描述信息，包括模型路径，推理runtime，输出结果的格式，输入输出层名称。这些内容都与模型强相关，在`yolov5s/YOLOv5s.cpp`的实现中，我把这一系列可配置的参数都开放出来了，以此增加代码对不同模型的适配能力。

```json
{
    "model-path":"../model/yolov5s.dlc",
    "runtime":"CPU",
    "labels":85,     // 5 + 检测类别数量
    "grids":25200,
    "input-layers":[ // 输入层名称
        "images"
    ],
    "output-layers":[ // 输出层名称
        "Sigmoid_199",
        "Sigmoid_201",
        "Sigmoid_203"
    ],
    "output-tensors":[ // 输出网络名称
        "output",
        "329",
        "331"
    ]
}
```

![1657616892233](images/1657616892233.png)

![1657356392597](images/1657356392597.png)

【注】：由于yolov5自身网络的原因，导致无法其无法在GPU runtime上获得正确的运行结果，因此不建议使用GPU runtime。

### test_video

`test_video`基于GStreamer框架，能够读取RTSP流并解码传递给Inference SDK进行推理，通过MQTT协议将推理结果发到MQTT Broker上，整体架构简图所示：

![Flowchart](images/Flowchart.jpg)

`VideoPipeline`：视频解码模块，使用`uridecodebin`插件，能够处理RTSP流或MP4视频文件。

`VideoAnalyzer`：视频分析模块，基于`libYOLOv5s.so`，完成推理并发送MQTT消息。

`SaftyQueue`：生产者消费者队列，`VideoPipeline`生产帧，`VideoAnalyzer`消耗帧。

```shell
# 运行test-video前确保MQTT Broker服务已经启动
# systemctl status mosquitto
./test/test_video/test-video --config_path ../test/test_video/config.json

# 新建终端，订阅test-topic即可看到推理结果
mosquitto_sub -t test-topic
```

一帧的推理结果输出的MQTT消息如下所示：

```json
{
   "results" : [
      {
         "bbox" : {
            "height" : 297,
            "width" : 225,
            "x" : 204,
            "y" : 366
         },
         "confidence" : 0.62800467014312744,
         "label" : "potted plant",
         "model" : "yolov5s-1"
      },
      {
         "bbox" : {
            "height" : 135,
            "width" : 39,
            "x" : 6,
            "y" : 606
         },
         "confidence" : 0.50076133012771606,
         "label" : "tv",
         "model" : "yolov5s-1"
      }
   ],
   "timestamp" : "1666162461677"
}
```

输入配置文件含义注释如下：

```json
{
    "pipeline-config":{
        "camera-url":"rtsp://admin:ZKCD1234@10.0.73.201:554",    // RTSP流URL
        "stream-width":1920,      // 输出图像尺寸
        "stream-height":1080,
        "output-format":"RGB",    // 输出图像格式
        "fps-n":25,    // 输出帧率控制参数，暂未支持
        "fps-d":1
    },
    "model-configs":[    // 模型配置，与test_image相同
        {
            "model-name":"yolov5s-1",
            "model-path":"../model/yolov5s.dlc",
            "label-path":"../model/yolov5s_labels.txt",
            "threshold-path":"../test/test_video/yolov5s_thresholds.txt",    // MQTT消息基于每个类的置信度过滤阈值，为80个float值（yolov5s支持80个类）
            "runtime":"DSP",
            "labels":85,
            "grids":25200,
            "input-layers":[
                "images"
            ],
            "output-layers":[
                "Sigmoid_199",
                "Sigmoid_201",
                "Sigmoid_203"
            ],
            "output-tensors":[
                "output",
                "329",
                "331"
            ],
            "global-threshold":0.2    // Inference SDK全局置信度阈值，必须比threshold-path中的值要小
        }
    ],
    "mqtt-config":{    // MQTT Client配置信息
        "topic-name":"test-topic",   // MQTT Topic
        "ip":"127.0.0.1",
        "port":1883,
        "keepalive":3,
        "QoS":1,
        "send-base64":false    // MQTT消息是否带Base64格式的图片，debug用
    }
}
```



