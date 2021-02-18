import socket
import struct


class TCPIP:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_adress = ("127.0.0.1", 10020)
        self.sock.connect(self.server_adress)
        self.format = "=ffffffdqdd" + "B"*14400  # format of struct
        # f =  float, d = double, q = longlong, B = uint8
        # 14400 = 60 x 60 x 3 

    def receive(self, MB_Sensors):
        data = self.sock.recv( 14456 ) # 56 byte + 14400 bytes
        unpacked = struct.unpack(self.format, data)
        MB_Sensors.setValues(unpacked)

    def send(self, MB_Motors):
        # MB_Motors.motor[0] = left motor , MB_Motors.motor[1] = right motor
        message = struct.pack("dd", MB_Motors.motor[0], MB_Motors.motor[1])
        self.sock.send(message)


    def close(self):
        self.sock.close()

