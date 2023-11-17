from threading import Thread

import imagezmq
import simplejpeg

from zmq2mjpeg.sensors import CatDetectSensor, DogDetectSensor, PersonDetectSensor


class CameraState:
    def __init__(self, name, last_frame=None):
        self.name = name
        self.last_frame = last_frame
        self.sensors = {
            "cat": CatDetectSensor(device_name=f"zmq2mjpeg_{name}"),
            "dog": DogDetectSensor(device_name=f"zmq2mjpeg_{name}"),
            "person": PersonDetectSensor(device_name=f"zmq2mjpeg_{name}")
        }


class CamTransformers:
    def __init__(self, plugins=None):
        if not plugins:  # TODO - OPM integration
            # from zmq2mjpeg.plugins.mobilenet import SSDLiteMobileNetV2
            # plugins = [SSDLiteMobileNetV2(valid_labels=["person", "cat", "dog"])]
            from zmq2mjpeg.plugins.yolo import YOLO
            plugins = [YOLO(valid_labels=["person", "cat", "dog"])]
        self.plugins = plugins

    def transform(self, frame, context=None):
        context = context or {}
        for p in self.plugins:
            frame, context = p.transform(frame, context)
        return frame, context


class CamReader(Thread):
    cameras = {}

    def __init__(self, daemon=True):
        super().__init__(daemon=daemon)
        self.image_hub = imagezmq.ImageHub()
        self.transformers = CamTransformers()

    def run(self) -> None:
        prev = None, None
        while True:

            rpi_name, jpg_buffer = self.image_hub.recv_jpg()
            frame = simplejpeg.decode_jpeg(jpg_buffer, colorspace='BGR')
            self.image_hub.send_reply(b'OK')
            if prev and (prev[1] == frame).all():
                continue  # no parsing repeat frames needed

            if rpi_name not in self.cameras:
                CamReader.cameras[rpi_name] = CameraState(rpi_name, frame)

            frame, _ = self.transformers.transform(frame, {"camera_name": rpi_name})

            CamReader.cameras[rpi_name].last_frame = frame
            prev = rpi_name, frame

    def get(self, name):
        cam = CamReader.cameras.get(name)
        if cam:
            return cam.last_frame
