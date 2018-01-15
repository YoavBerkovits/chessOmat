import socket
import cv2
from threading import Thread

LAPTOP_IP = '192.168.43.167'
PORT = 5000
TIMEOUT = 1.0 #seconds

class sender:

    def __init__(self):

        def setup_socket(me):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(TIMEOUT)
                s.connect((LAPTOP_IP, PORT))
                print("Connected to GUI!")
                self.socket = s
            except:
                print("Failed to connect to GUI!")

        self.thread = Thread(target=setup_socket, args=(self,))
        self.thread.start()


    def send_image(self, img):
        def really_send(img):
            try:
                str_encode = cv2.imencode('.jpg', img)[1].tostring()
                self.socket.send(str_encode)
            except:
                print("failed to send image <:-(")

        if not self.thread==None:
            self.thread.join()

        self.thread = Thread(target = really_send, args = (img,))
        self.thread.start()

    def send_msg(self, msg):

        self.socket.send(msg.encode())
