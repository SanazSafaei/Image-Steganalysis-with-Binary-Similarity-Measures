import cv2
import numpy as np
import math
import cmath


class Measures:

    def __init__(self, org, final):
        self.org = cv2.imread(org, cv2.IMREAD_GRAYSCALE).astype('int64')
        self.final = cv2.imread(final, cv2.IMREAD_GRAYSCALE).astype('int64')
        self.size = self.org.shape
        self.imageQualityMetrics = []
        self.MeanError()
        self.MeanSquareError()
        self.CzekanowskiCorrelationMeasure()
        self.ImageFidelity()
        self.SpectralMagnitudeDistance()
        self.MedianBlockSpectralPhaseDistance()

    def MeanError(self):
        sub = self.final - self.org  # submission of each index of changed picture and original image
        sum_pixels = np.sum(sub[:, :])  # Sum of each index of matrix
        avg = np.divide(sum_pixels, (self.size[0] * self.size[1]))
        mse = np.sum(avg)
        # result = self.psnre(mse)
        print("Mean absolute error : ", mse)
        self.imageQualityMetrics.append(mse)

    def MeanSquareError(self):
        sub = self.org - self.final  # submission of each index of changed picture and original image
        pow_sub = np.power(sub, 2)  # each index of matrix to power of 2
        sum_pixels = np.sum(pow_sub[:, :])  # Sum of each index of matrix
        avg = sum_pixels / (self.size[0] * self.size[1])
        result = math.sqrt(avg)
        # result = self.psnre(mse)
        print("Mean square error : ", result)
        self.imageQualityMetrics.append(result)

    # def psnre(self, result):
    #     if (result > 0):
    #         psnre = 10 * math.log((255 * 255 / result), 10)
    #     else:
    #         psnre = 99
    #     print("psnre : ", result)
    #     return psnre

    def CzekanowskiCorrelationMeasure(self):
        np.seterr(divide='ignore', invalid='ignore')  # Catching error in case of dividing 0 to 0
        minimum = cv2.min(self.org, self.final)  # finding the minimum of eac index between two images matrix
        sum = self.org + self.final  # sum of two matrix
        minimum2 = minimum * 2
        # for i in range(0,self.size[0]-1):
        #     for j in range(0,self.size[1]-1):
        #         if (sum[i,j]==0):
        #             print("this ->", minimum[i,j],"____", self.org[i,j], "___",self.final[i,j])
        d = np.divide(minimum2, sum)
        d[np.isnan(d)] = 1  # replacing Nan to 1 in case of dividing 0 to 0
        # d[np.isinf(d)] = 0
        n2 = (self.size[0] * self.size[1]) ** 2
        temp = 1 - d
        s = np.sum(temp[:, :])
        result = s / n2
        print("Czekanowski Correlation Measure : ", result)
        self.imageQualityMetrics.append(result)

    def ImageFidelity(self):
        sub = self.org - self.final  # submission of each index of changed picture and original image
        sub2 = sub ** 2  # each index of matrix to power of 2
        sub2 = np.sum(sub2[:, :])
        dementor = np.sum(self.org ** 2)
        d = sub2 / dementor
        result = 1 - d
        print("Cross correlation : ", d)
        print("Image fidelity : ", result)
        self.imageQualityMetrics.append(d)
        self.imageQualityMetrics.append(result)

    # def AngelMean(self): #it can't be use for grayscale images cause the answer is always 1
    #     N = self.size[0] * self.size[1]
    #     matrix_multiplication = self.org * self.final  # Multiplication of each index of matrices
    #     sum = np.sum(matrix_multiplication)
    #     c1 = np.power(self.org, 2)
    #     sum_c1 = np.sum(c1)
    #     sqrt_c1 = math.sqrt(sum_c1)
    #     c2 = np.power(self.org, 2)
    #     sum_c2 = np.sum(c2)
    #     sqrt_c2 = math.sqrt(sum_c2)
    #     d = sum / sqrt_c1 * sqrt_c2
    #     d = min(1, max(d, -1))  # replacing Nan to 1 in case of dividing 0 to 0
    #     result = (math.acos(d) * 2) / math.pi
    #     result = 1 - (result / N ** 2)
    #     print("Angel Mean : ", result)
    #     self.imageQualityMetrics.append(result)

    # def DFT(self, u, v, img): //need to optimise
    #     N = self.size[0] * self.size[1]
    #     sum = 0
    #     for m in range(0, self.size[1]):
    #         for n in range(0, self.size[0]):
    #             temp1 = math.pi * -2 * m * u / N
    #             c1 = cmath.exp(temp1 * 1j)
    #             temp2 = math.pi * -2 * n * v / N
    #             c2 = cmath.exp(temp2 * 1j)
    #             sum = sum + c1 * c2 * img[m][n]
    #     return sum

    def SpectralMagnitudeDistance(self):
        N2 = (self.size[0] * self.size[1]) ** 2
        gamma1 = np.abs(np.fft.fft2(self.org))  # calling Discrete Fourier Transforms 2-dimensional for original image
        gamma2 = np.abs(np.fft.fft2(self.final))  # Discrete Fourier Transforms 2-dimensional for changed image
        temp = (gamma1 - gamma2) ** 2
        sum = np.sum(temp)
        result = sum / N2

        print("Spectral Magnitude Distance : ", result)
        self.imageQualityMetrics.append(result)

    def j_measure(self, matris_orgin, matris_final):
        fourier_orgin = np.angle(np.fft.fft2(matris_orgin))  # abs calculates magnitude
        fourier_final = np.angle(np.fft.fft2(matris_final))
        sub = fourier_orgin - fourier_final
        # print("sub ", sub)
        sum = np.sum(sub ** 2)
        # print("sum ", sum)
        result = np.sqrt(sum)
        return result

    def MedianBlockSpectralPhaseDistance(self):
        N = self.size[0] * self.size[1]
        print("L len",N/(32*32))
        print(self.size)
        if self.size[0] % 32 == 0 and self.size[1] % 32 == 0:
            L = np.empty(int(N / (32 * 32)))
            for j in range(0, self.size[1], 32):
                for i in range(0, self.size[0], 32):
                    np.append(L, self.j_measure(self.org[i:i + 32, j:j + 32], self.final[i:i + 32, j:j + 32]))

        elif self.size[0] % 30 == 0 and self.size[1] % 30 == 0:
            L = np.empty(int(N / (30 * 30)))
            for j in range(0, self.size[1]-1, 30):
                for i in range(0, self.size[0]-1, 30):
                    np.append(L, self.j_measure(self.org[i:i + 30, j:j + 30], self.final[i:i + 30, j:j + 30]))

        elif self.size[0] % 31 == 0 or self.size[1] % 31 == 0:
            L = np.empty(int(N / (30 * 32)))
            for j in range(0, self.size[1], 6):
                for i in range(0, self.size[0], 6):
                    np.append(L, self.j_measure(self.org[i:i + 16, j:j + 16], self.final[i:i + 16, j:j + 16]))

        else:
            L=np.zeros(2)
            print("None")
        # print("L : ", np.sort(L))
        L = np.nan_to_num(L)
        result = np.median(L)
        print("Median Block Spectral Phase Distance : ", result)
        self.imageQualityMetrics.append(result)

    def get_metrics(self):
        result = np.array(self.imageQualityMetrics)
        result=result.ravel()
        return result


