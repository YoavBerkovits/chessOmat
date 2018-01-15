import socket
import cv2
import numpy

PORT = 5000
IM_SIZE = 819200
TIMEOUT = 15
class listener:

    def __init__(self):
        self.setup_socket()
        self.setup_connection()

    def setup_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", PORT))
        self.sock = s
        self.sock.settimeout(TIMEOUT)


    def setup_connection(self):
        print('listennin')
        self.sock.listen(1)
        sender, address = self.sock.accept()
        print("Successfully connected to pi: ", address)
        self.connection = sender

    def get_image(self):
        while True:
            msg = self.connection.recv(IM_SIZE)
            decoded = numpy.fromstring(msg,numpy.uint8)
            img = cv2.imdecode(decoded,
                                cv2.IMREAD_COLOR)
            if not (img is None):
                return img

    def get_turn(self):
        return self.connection.recv(IM_SIZE).decode()

    def close(self):
        self.connection.close()
        self.sock.close()


    def send_results(self):
        self.connection.send()

