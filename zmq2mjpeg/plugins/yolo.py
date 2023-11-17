import colorsys
import random

import cv2
from ovos_PHAL_sensors.loggers import HomeAssistantUpdater

from zmq2mjpeg.cam import CamReader
from zmq2mjpeg.plugins.cvlib_yolo import detect_common_objects, classes


class YOLO:
    def __init__(self, valid_labels=None, min_score=0.7, publish_sensors=True, n_frames=3):
        self.min_score = min_score
        self.valid_labels = valid_labels or []

        # Generate colors for drawing bounding boxes.
        self.colors = self.generate_colors(class_names=classes)

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

    def draw_boxes(self, image, out_scores, out_boxes, out_classes):
        h, w, _ = image.shape

        for predicted_class, score, box in zip(out_classes, out_scores, out_boxes):
            if self.valid_labels and predicted_class not in self.valid_labels:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)

            left, top, right, bottom = box

            # colors: RGB, opencv: BGR
            cv2.rectangle(image, (left, top), (right, bottom),
                          tuple(reversed(self.colors[classes.index(predicted_class)])), 6)

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
            label_rect_left, label_rect_top = int(left - 3), int(top - 3)
            label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
            cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                          tuple(reversed(self.colors[classes.index(predicted_class)])), -1)

            cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)

        return image

    def detect_frame(self, rpi_name, frame, draw=False):
        out_boxes, out_classes, out_scores = detect_common_objects(frame, confidence=0.25,
                                                                   model='yolov3-tiny')
        detections = {}
        for predicted_class, score in zip(out_classes, out_scores):
            if self.valid_labels and predicted_class not in self.valid_labels:
                continue
            if score < self.min_score:
                continue
            detections[predicted_class] = score

        # print(rpi_name, detections)
        cv2.imshow(rpi_name, frame)
        cv2.waitKey(1)
        if draw:  # modifies frame object
            self.draw_boxes(frame, out_scores, out_boxes, out_classes)
        cv2.imshow(rpi_name + "_detected", frame)
        cv2.waitKey(1)

        self._process_detections(rpi_name, detections)
        return detections

    def transform(self, frame, context):
        name = context["camera_name"]
        draw = context.get("draw", True)
        preds = self.detect_frame(name, frame, draw=draw)
        context["yolov3-tiny"] = preds
        return frame, context
