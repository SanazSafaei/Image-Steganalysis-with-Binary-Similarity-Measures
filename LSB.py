import cv2
import math
import copy


class LSB:

    def __init__(self, msg, img, path):
        self.msg = msg
        self.img = img
        self.path = path

    def msgToBinary(self):
        stringBinary = ' '.join(map(bin, bytearray(self.msg, 'utf8')))
        stringBinary = stringBinary.replace(" ", "")  # attaching words together
        stringBinary = stringBinary.replace("0b", "")  # removing 0b from string
        return stringBinary

    def decimalToBinary(self, number):
        numBinary = bin(number)
        numBinary = numBinary.replace("0b", "")  # removing 0b from string
        numBinary = list(numBinary)
        return numBinary

    def replaceBinary(self, a, b):
        result = a.copy()
        result[-1] = b
        return result

    def BinarytoDecimal(self, a):
        binary = "".join(a)
        decimal = int(binary, 2)
        return decimal

    def lsb(self):
        secretMsg = self.msgToBinary()
        msg_counter = len(secretMsg)-1
        I = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE).astype('int64')
        I2 = copy.deepcopy(I)
        height, width = I.shape
        for i in range(0, height):
            if (msg_counter == -1):
                break
            for j in range(0, width):
                if (msg_counter == -1):
                    break
                bit = self.decimalToBinary(I[i][j])
                bit2 = self.replaceBinary(bit, secretMsg[msg_counter])
                I2[i][j] = self.BinarytoDecimal(bit2)
                msg_counter -= 1

        I2 = cv2.normalize(I2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        name = self.path + "/lsb-" + str(len(self.msg)) + "-" + self.img[9:]
        cv2.imwrite(name, I2)

