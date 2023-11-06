from threading import Thread

import imagezmq

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
            from zmq2mjpeg.plugins.mobilenet import SSDLiteMobileNetV2
            plugins = [SSDLiteMobileNetV2(valid_labels=["person", "cat", "dog"])]
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
        while True:

            rpi_name, frame = self.image_hub.recv_image()
            self.image_hub.send_reply(b'OK')

            if rpi_name not in self.cameras:
                self.cameras[rpi_name] = CameraState(rpi_name, frame)

            frame, _ = self.transformers.transform(frame, {"camera_name": rpi_name})

            self.cameras[rpi_name].last_frame = frame

    def get(self, name):
        cam = self.cameras.get(name)
        if cam:
            return cam.last_frame
