import dataclasses

from ovos_PHAL_sensors.sensors.base import BooleanSensor


@dataclasses.dataclass
class PersonDetectSensor(BooleanSensor):
    unique_id: str = "person"
    device_name: str = "zmq2mjpeg"
    detected: bool = False
    confidence: float = 0.0

    @property
    def value(self):
        return self.detected


@dataclasses.dataclass
class DogDetectSensor(BooleanSensor):
    unique_id: str = "dog"
    device_name: str = "zmq2mjpeg"
    detected: bool = False
    confidence: float = 0.0

    @property
    def value(self):
        return self.detected


@dataclasses.dataclass
class CatDetectSensor(BooleanSensor):
    unique_id: str = "cat"
    device_name: str = "zmq2mjpeg"
    detected: bool = False
    confidence: float = 0.0

    @property
    def value(self):
        return self.detected
