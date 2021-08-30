import glob
import random
import numpy as np
import cv2


def gaussian_noise(path,sig, num, start):
    fnames_original = glob.glob('original/*.*')

    # if path == "train/stego":
    idx = fnames_original[start:start+num]
    # else:
    #     idx = [random.randint(0, len(fnames_original) - 1) for i in range(40)]
    for i in idx:
        I = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

        I = I.astype(np.float) / 255

        # create the noise image
        sigma = sig  # notice maximum intensity is 1
        N = np.random.randn(*I.shape) * sigma

        # add noise to the original image
        J = I + N  # or use cv2.add(I,N);
        result = cv2.normalize(J, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # if path == "train/stego":
        name = "noise/" + i[9:]
        # else:
        #     name = "" + path + "/noise/" + fnames_original[i][14:]
        cv2.imwrite(name, result)
