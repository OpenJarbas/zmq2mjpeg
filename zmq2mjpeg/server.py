from threading import Thread

import cv2
import imagezmq
from flask import Flask, Response
from ovos_PHAL_sensors.loggers import HomeAssistantUpdater

from zmq2mjpeg.sensors import CatDetectSensor, DogDetectSensor, PersonDetectSensor

# from argparse TODO
DETECT = True
MODEL_PATH = "ssdlite_mobilenet_v2.tflite"
VALID_LABELS = ["person", "cat", "dog"]
N_FRAMES = 3  # consecutive frame detections to count as valid detection
PUBLISH_SENSORS = True

HA_URL = "http://192.168.1.8:8123"
HA_TOKEN = "ey5MTMxNDgwODAyMmRmMiIs..."


class CameraState:
    def __init__(self, name, last_frame=None):
        self.name = name
        self.last_frame = last_frame
        self.sensors = {
            "cat": CatDetectSensor(device_name=f"zmq2mjpeg_{name}"),
            "dog": DogDetectSensor(device_name=f"zmq2mjpeg_{name}"),
            "person": PersonDetectSensor(device_name=f"zmq2mjpeg_{name}")
        }


class CamReader(Thread):
    image_hub = imagezmq.ImageHub()
    cameras = {}
    if DETECT:
        from zmq2mjpeg.objdetect import ObjDetect
        detector = ObjDetect(MODEL_PATH, VALID_LABELS)
    else:
        detector = None
    _COUNTS = {l: 0 for l in VALID_LABELS}

    def _process_detections(self, name, preds):
        detections = list(preds.keys())
        for label, conf in preds.items():
            # print(label, conf, self._COUNTS[label])
            if self._COUNTS[label] <= N_FRAMES:
                self._COUNTS[label] += 1
            if self._COUNTS[label] >= N_FRAMES:
                if not self.cameras[name].sensors[label].detected:
                    print("DETECTED:", label, conf)
                    self.cameras[name].sensors[label].detected = True
                    if PUBLISH_SENSORS:
                        HomeAssistantUpdater.binary_sensor_update(self.cameras[name].sensors[label])
                self.cameras[name].sensors[label].confidence = conf
        for label in self._COUNTS:
            if label not in detections:
                if self._COUNTS[label] > 0:
                    self._COUNTS[label] -= 1
                    self.cameras[name].sensors[label].confidence -= 0.1
                if self._COUNTS[label] <= 0 and self.cameras[name].sensors[label].detected:
                    print("LOST DETECTION:", label)
                    self.cameras[name].sensors[label].detected = False
                    if PUBLISH_SENSORS:
                        HomeAssistantUpdater.binary_sensor_update(self.cameras[name].sensors[label])
                self.cameras[name].sensors[label].confidence = 0

    def run(self) -> None:
        while True:

            rpi_name, frame = self.image_hub.recv_image()
            self.image_hub.send_reply(b'OK')

            if rpi_name not in self.cameras:
                self.cameras[rpi_name] = CameraState(rpi_name, frame)

            if self.detector:
                preds = self.detector.detect_frame(frame, draw=True)
                self._process_detections(rpi_name, preds)

            self.cameras[rpi_name].last_frame = frame

    def get(self, name):
        cam = self.cameras.get(name)
        if cam:
            return cam.last_frame


def get_app():
    app = Flask(__name__)

    HomeAssistantUpdater.ha_url = HA_URL
    HomeAssistantUpdater.ha_token = HA_TOKEN

    image_hub = CamReader(daemon=True)
    image_hub.start()

    def _gen_frames(target_name):  # generate frame by frame from camera
        while True:
            frame = image_hub.get(target_name)
            if frame is None:
                continue
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                pass

    @app.route('/video_feed/<name>')
    def video_feed(name):
        return Response(_gen_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def main():
    app = get_app()
    app.run(host="0.0.0.0")


if __name__ == '__main__':
    main()
