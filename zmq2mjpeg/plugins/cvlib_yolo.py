# taken from https://github.com/arunponnusamy/cvlib
# not added to requirements.txt to not drag tensorflow

import os

import cv2
import numpy as np

net = None
dest_dir = os.path.expanduser(
    '~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
           'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
           'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable',
           'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def detect_common_objects(image, confidence=0.5, nms_thresh=0.3, model='yolov4', enable_gpu=False):
    """A method to detect common objects
    Args:
        image: A colour image in a numpy array
        confidence: A value to filter out objects recognised to a lower confidence score
        nms_thresh: An NMS value
        model: The detection model to be used, supported models are: yolov3, yolov3-tiny, yolov4, yolov4-tiny
        enable_gpu: A boolean to set whether the GPU will be used

    """

    global classes, net, dest_dir
    Height, Width = image.shape[:2]
    scale = 0.00392
    config_file_name = 'yolov3-tiny.cfg'
    weights_file_name = 'yolov3-tiny.weights'

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    config_file_abs_path = f"{os.path.dirname(__file__)}/{config_file_name}"
    weights_file_abs_path = f"{os.path.dirname(__file__)}/{weights_file_name}"

    if net is None:
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)

    # enables opencv dnn module to use CUDA on Nvidia card instead of cpu
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox.append([int(x), int(y), int(x + w), int(y + h)])
        label.append(str(classes[class_ids[i]]))
        conf.append(confidences[i])

    return bbox, label, conf
