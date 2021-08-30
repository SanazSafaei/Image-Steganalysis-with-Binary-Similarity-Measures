import cv2
import math
import copy


class PVD:

    def __init__(self, msg, img,path):
        self.msg = msg
        self.img = img
        self.path=path

    def msgToBinary(self):
        stringBinary = ' '.join(map(bin, bytearray(self.msg, 'utf8')))
        stringBinary = stringBinary.replace(" ", "")  # attaching words together
        stringBinary = stringBinary.replace("0b", "")  # removing 0b from string
        return stringBinary

    def range_table(self, d):
        R = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]  # pvd reference table
        for i in R:
            if i[0] <= d:
                if i[1] >= d:
                    bits = int(math.log2(i[1] - i[0]))  # finding the matching range
                    return i[0], bits

    def pvd(self):
        secretMsg = self.msgToBinary()
        msg_counter = 0
        I = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE).astype('int64')
        I2 = copy.deepcopy(I)
        height, width = I.shape
        flag = False
        p1 = 0
        for h in range(0, height):  # iterating in image height
            if msg_counter == len(secretMsg):  # if the message is over
                break
            if h % 2 == 0:  # if row is even we should iterate at the first of row
                iterate_w = range(0, width, 1)
            else:  # if row is odd we should iterate at the end of row
                iterate_w = range(width - 1, -1, -1)
            for w in iterate_w:  # iterating in image width
                if msg_counter >= len(secretMsg):
                    break
                if flag:  # if p1 refreshed
                    p0 = I[h][w]  # switching pixel pairs (old p1 goes to new p0 and new p1 adds)
                    d = p0 - p1
                    d = abs(d)
                    temp = self.range_table(d)
                    if msg_counter + temp[1] <= len(
                            secretMsg):  # if the remaining message is bigger than the chosen bits
                        msg_temp = int(secretMsg[msg_counter:msg_counter + temp[1]], 2)
                    elif msg_counter + temp[1] >= len(
                            secretMsg) > msg_counter:  # if the remaining message is less than the chosen bits, put all the remaining message
                        msg_temp = int(secretMsg[msg_counter:])
                    msg_counter = msg_counter + temp[1]
                    if msg_counter >= len(secretMsg) - 1:  # if the message is over
                        break
                    d2 = math.fabs(temp[0] + msg_temp)
                    m = math.fabs(d - d2)
                    if d % 2 == 1:
                        I2[h][w] = p1 + math.floor(m / 2)
                        if h % 2 == 1 and h == 0:
                            I2[h - 1][width - 1] = p0 - math.ceil(m / 2)
                        else:
                            I2[h][w - 1] = p0 - math.ceil(m / 2)
                    else:
                        I2[h][w] = p1 + math.ceil(m / 2)
                        if h % 2 == 1 and h == 0:
                            I2[h - 1][width - 1] = p0 - math.floor(m / 2)
                        else:
                            I2[h][w - 1] = p0 - math.floor(m / 2)
                else:  # next pixel
                    p1 = I[h][w]
                flag = not flag

        I = cv2.normalize(I, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        I2 = cv2.normalize(I2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # if self.path=="train/stego":
        name = self.path+"/pvd-" + str(len(self.msg)) + "-" + self.img[9:]
        # else:
        #     name = self.path + "/pvd-" + str(len(self.msg)) + "-" + self.img[14:]
        cv2.imwrite(name, I2)
