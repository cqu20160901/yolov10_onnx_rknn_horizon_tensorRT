import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt
import time

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)



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
object_thresh = 0.4

input_height = 640
input_width = 640

topK = 50


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


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


def preprocess(src_image, input_width, input_height):
    image = cv2.resize(src_image, (input_width, input_height)).astype(np.float32)
    image = image * 0.00392156
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    return image


def main():
    engine_file_path = './yolov10n_zq.trt'
    input_image_path = './test.jpg'

    orign_image = cv2.imread(input_image_path)
    image = cv2.cvtColor(orign_image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    image = preprocess(image, input_width, input_height)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
        t1  = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        t2 = time.time()
        print('run tiems time:', (t2 - t1))

        print('outputs heads num: ', len(trt_outputs))

        out = []
        for i in range(len(trt_outputs)):
            out.append(trt_outputs[i])

        predbox = postprocess(out, img_h, img_w)

        print(len(predbox))

        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score

            cv2.rectangle(orign_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            title = CLASSES[classId] + "%.2f" % score
            cv2.putText(orign_image, title, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite('./test_result_tensorRT.jpg', orign_image)


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()
    main()
