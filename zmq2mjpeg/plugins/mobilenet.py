# adapted from https://github.com/kaka-lin/object-detection
# MIT License

import colorsys
import os
import random

import cv2
import numpy as np
from ovos_PHAL_sensors.loggers import HomeAssistantUpdater

from zmq2mjpeg.cam import CamReader

try:
    import tensorflow.lite as tflite
except:
    import tflite_runtime.interpreter as tflite


class SSDLiteMobileNetV2:
    def __init__(self, valid_labels=None, model_path=None, min_score=0.7, publish_sensors=True, n_frames=3):
        self.min_score = min_score
        self.valid_labels = valid_labels or []
        # Load TFLite model and allocate tensors.
        model_path = model_path or f"{os.path.dirname(__file__)}/ssdlite_mobilenet_v2.tflite"
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # label
        self.class_names = ['None1', 'person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck',
                            'boat', 'traffic light', 'fire hydrant', 'None12', 'stop sign', 'parking meter', 'bench',
                            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                            'None', 'backpack', 'umbrella', 'None29', 'Noe30', 'handbag', 'tie', 'suitcase', 'frisbee',
                            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'None45', 'wine glass', 'cup', 'fork', 'knife',
                            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                            'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'None66', 'diningtable',
                            'None68', 'None69', 'toilet', 'None71', 'tvmonitor', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'None83',
                            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # Generate colors for drawing bounding boxes.
        self.colors = self.generate_colors(self.class_names)

        self.publish_sensors = publish_sensors
        self.n_frames = n_frames
        self._COUNTS = {l: 0 for l in self.valid_labels}

    def _process_detections(self, name, preds):
        detections = list(preds.keys())
        for label, conf in preds.items():
            # print(label, conf, self._COUNTS[label])
            if self._COUNTS[label] <= self.n_frames:
                self._COUNTS[label] += 1
            if self._COUNTS[label] >= self.n_frames:
                if not CamReader.cameras[name].sensors[label].detected:
                    print("DETECTED:", label, conf)
                    CamReader.cameras[name].sensors[label].detected = True
                    if self.publish_sensors:
                        HomeAssistantUpdater.binary_sensor_update(CamReader.cameras[name].sensors[label])
                CamReader.cameras[name].sensors[label].confidence = conf
        for label in self._COUNTS:
            if label not in detections:
                if self._COUNTS[label] > 0:
                    self._COUNTS[label] -= 1
                    CamReader.cameras[name].sensors[label].confidence -= 0.1
                if self._COUNTS[label] <= 0 and CamReader.cameras[name].sensors[label].detected:
                    print("LOST DETECTION:", label)
                    CamReader.cameras[name].sensors[label].detected = False
                    if self.publish_sensors:
                        HomeAssistantUpdater.binary_sensor_update(CamReader.cameras[name].sensors[label])
                CamReader.cameras[name].sensors[label].confidence = 0

    @staticmethod
    def generate_colors(class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    @staticmethod
    def preprocess_image_for_tflite(image, model_image_size=300):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (model_image_size, model_image_size))
        image = np.expand_dims(image, axis=0)
        image = (2.0 / 255.0) * image - 1.0
        image = image.astype('float32')
        return image

    @staticmethod
    def non_max_suppression(scores, boxes, classes, max_boxes=10, min_score_thresh=0.5):
        out_boxes = []
        out_scores = []
        out_classes = []
        if not max_boxes:
            max_boxes = boxes.shape[0]
        for i in range(min(max_boxes, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                out_boxes.append(boxes[i])
                out_scores.append(scores[i])
                out_classes.append(classes[i])

        out_boxes = np.array(out_boxes)
        out_scores = np.array(out_scores)
        out_classes = np.array(out_classes)

        return out_scores, out_boxes, out_classes

    def draw_boxes(self, image, out_scores, out_boxes, out_classes):
        h, w, _ = image.shape

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if self.valid_labels and predicted_class not in self.valid_labels:
                continue
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (xmin * w, xmax * w,
                                        ymin * h, ymax * h)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
            right = min(w, np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # colors: RGB, opencv: BGR
            cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(self.colors[c])), 6)

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
            label_rect_left, label_rect_top = int(left - 3), int(top - 3)
            label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
            cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                          tuple(reversed(self.colors[c])), -1)

            cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)

        return image

    def run_detection(self, image):
        # Run model: start to detect
        # Sets the value of the input tensor.
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        # Invoke the interpreter.
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])

        boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
        out_scores, out_boxes, out_classes = self.non_max_suppression(scores, boxes, classes)

        return out_scores, out_boxes, out_classes

    def detect_frame(self, rpi_name, frame, draw=False):
        image_data = self.preprocess_image_for_tflite(frame, model_image_size=300)
        out_scores, out_boxes, out_classes = self.run_detection(image_data)
        detections = {}
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if self.valid_labels and predicted_class not in self.valid_labels:
                continue
            score = out_scores[i]
            if score < self.min_score:
                continue
            detections[predicted_class] = score

        if draw:  # modifies frame object
            self.draw_boxes(frame, out_scores, out_boxes, out_classes)

        self._process_detections(rpi_name, detections)
        return detections

    def transform(self, frame, context):
        name = context["camera_name"]
        draw = context.get("draw", True)
        preds = self.detect_frame(name, frame, draw=draw)
        context["ssdlite_mobilenet_v2"] = preds
        return frame, context
