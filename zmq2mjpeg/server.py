import cv2
from flask import Flask, Response
from ovos_PHAL_sensors.loggers import HomeAssistantUpdater

from zmq2mjpeg.cam import CamReader


HA_URL = "http://192.168.1.8:8123"
HA_TOKEN = "ey5MTMxNDgwODAyMmRmMiIs..."


def get_app():
    app = Flask(__name__)

    HomeAssistantUpdater.ha_url = HA_URL
    HomeAssistantUpdater.ha_token = HA_TOKEN

    image_hub = CamReader()
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
