from time import sleep

import imagezmq
import zmq  # needed because we will be using zmq socket options & exceptions
from imutils.video import VideoStream

connect_to = 'tcp://0.0.0.0:5555'  # replace ip with the zmq2mjpeg host ip
rpi_name = "pc_do_miro"  # some unique device name here
time_between_restarts = 5  # number of seconds to sleep between sender restarts


def sender_start(connect_to=None):
    sender = imagezmq.ImageSender(connect_to=connect_to)
    sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    # NOTE: because of the way PyZMQ and imageZMQ are implemented, the
    #       timeout values specified must be integer constants, not variables.
    #       The timeout value is in milliseconds, e.g., 2000 = 2 seconds.
    sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 2000)  # set a receive timeout
    sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 2000)  # set a send timeout
    return sender



def main():
    sender = sender_start(connect_to)

    picam = VideoStream().start()

    try:
        while True:  # send images as stream until Ctrl-C
            image = picam.read()
            try:
                reply_from_mac = sender.send_image(rpi_name, image)
            except (zmq.ZMQError, zmq.ContextTerminated, zmq.Again):
                if 'sender' in locals():
                    print('Closing ImageSender.')
                    sender.close()
                sleep(time_between_restarts)
                print('Restarting ImageSender.')
                sender = sender_start(connect_to)
    except (KeyboardInterrupt, SystemExit):
        pass  # Ctrl-C was pressed to end program
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
    finally:
        if 'sender' in locals():
            sender.close()
        picam.stop()  # stop the camera thread


if __name__ == "__nain__":
    main()