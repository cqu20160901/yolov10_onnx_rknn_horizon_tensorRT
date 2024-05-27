import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './yolov10n_zq.onnx'
RKNN_MODEL = './yolov10n_zq.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

meshgrid = []

class_num = len(CLASSES)
head_num = 3
strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]
object_thresh = 0.3

input_height = 640
input_width = 640

topK = 50


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def TopK(detectResult):
    if len(detectResult) <= topK:
        return detectResult
    else:
        predBoxs = []
        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
        for i in range(topK):
            predBoxs.append(sort_detectboxs[i])
        return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    scale_h = img_h / input_height
    scale_w = img_w / input_width

    gridIndex = -2
    cls_index = 0
    cls_max = 0

    for index in range(head_num):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]

        for h in range(map_size[index][0]):
            for w in range(map_size[index][1]):
                gridIndex += 2

                if 1 == class_num:
                    cls_max = sigmoid(cls[0 * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                    cls_index = 0
                else:
                    for cl in range(class_num):
                        cls_val = cls[cl * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w]
                        if 0 == cl:
                            cls_max = cls_val
                            cls_index = cl
                        else:
                            if cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                    cls_max = sigmoid(cls_max)

                if cls_max > object_thresh:
                    regdfl = []
                    for lc in range(4):
                        sfsum = 0
                        locval = 0
                        for df in range(16):
                            temp = exp(reg[((lc * 16) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                            reg[((lc * 16) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][ 1] + w] = temp
                            sfsum += temp

                        for df in range(16):
                            sfval = reg[((lc * 16) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][
                                1] + w] / sfsum
                            locval += sfval * df
                        regdfl.append(locval)

                    x1 = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index]
                    y1 = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index]
                    x2 = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index]
                    y2 = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index]

                    xmin = x1 * scale_w
                    ymin = y1 * scale_h
                    xmax = x2 * scale_w
                    ymax = y2 * scale_h

                    xmin = xmin if xmin > 0 else 0
                    ymin = ymin if ymin > 0 else 0
                    xmax = xmax if xmax < img_w else img_w
                    ymax = ymax if ymax < img_h else img_h

                    box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                    detectResult.append(box)
    # topK
    print('before topK num is:', len(detectResult))
    predBox = TopK(detectResult)

    return predBox



def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()

    img_path = './test.jpg'
    src_img = cv2.imread(img_path)
    img_h, img_w = src_img.shape[:2]
    
    
    input_img = cv2.resize(src_img, (input_width, input_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    input_img = np.expand_dims(input_img, 0)

    outputs = export_rknn_inference(input_img)

    out = []
    for i in range(len(outputs)):
        out.append(outputs[i])

    predbox = postprocess(out, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        title = CLASSES[classId] + ":%.2f" % (score)
        cv2.putText(src_img, title, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_rknn_result.jpg', src_img)

