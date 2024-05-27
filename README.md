# yolov10_onnx_rknn_horizon_tensorRT

yolov10 目标检测部署版本，便于移植不同平台（onnx、tensorRT、rknn、Horizon），全网部署最简单、运行速度最快的部署方式（全网首发），后处理为C++部署而写，python 测试后处理时耗意义不大。

yolov10_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov10_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolov10_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

yolov10_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

**导出onnx参考链接[【yolov10 瑞芯微RKNN、地平线Horizon芯片部署、TensorRT部署，部署工程难度小、模型推理速度快】](https://blog.csdn.net/zhangqian_1/article/details/139239964)**

# 测试效果

pytorch 结果

![test](https://github.com/cqu20160901/yolov10_onnx_rknn_horizon_tensorRT/assets/22290931/d7eaa71b-2c78-4f9a-acf0-1a0d08aba6e6)

onnx 结果

![image](https://github.com/cqu20160901/yolov10_onnx_rknn_horizon_tensorRT/blob/main/yolov10_onnx/test_onnx_result.jpg)

# 时耗

tensorRT部署推理时耗（显卡 Tesla V100、cuda_11.0）

![image](https://github.com/cqu20160901/yolov10_onnx_rknn_horizon_tensorRT/assets/22290931/3ded60fd-aa4a-4012-b54a-5c83afd17e14)


rk3588 时耗（模型输入640x640，检测80类）

![image](https://github.com/cqu20160901/yolov10_onnx_rknn_horizon_tensorRT/assets/22290931/8accaafb-2b3e-45f2-a4cf-986cc72f35fd)

本示例用的是yolov10n，模型计算量6.7G，看到这个时耗觉得可能是有操作切换到CPU上进行计算的，查了rknn转换模型日志确实是有操作切换到CPU上进行的，对应的是模型中 PSA 模块计算 Attention 这部分操作。

![image](https://github.com/cqu20160901/yolov10_onnx_rknn_horizon_tensorRT/assets/22290931/ab42ab0b-cda0-43a1-9666-a59f908fbae8)


# rknn 部署C++代码

[C++代码](https://github.com/cqu20160901/yolov10_rknn_Cplusplus)






