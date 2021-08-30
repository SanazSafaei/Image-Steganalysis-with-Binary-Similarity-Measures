import random
import pixelValueDifferencing
import glob
import LSB
import Gaussian_noise

messages = glob.glob('texts/*.txt')
original = glob.glob('original/*.*')
train_original = original[-20:]   # all of the original images in folder images in type of png

for msg_name in messages:

    train_f = open(msg_name, "r")
    train_msg = train_f.read()

    for train_img in train_original:
        p = pixelValueDifferencing.PVD(train_msg, train_img, 'stego')
        p.pvd()
        lsb = LSB.LSB(train_msg, train_img, 'stego')
        lsb.lsb()


Gaussian_noise.gaussian_noise('train',0.02,50,15)
Gaussian_noise.gaussian_noise('train',0.5,5,66)
Gaussian_noise.gaussian_noise('train',0.1,5,72)
# Gaussian_noise.gaussian_noise('test')

print("DONE!")
