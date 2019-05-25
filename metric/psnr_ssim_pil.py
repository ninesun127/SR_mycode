#coding=utf-8

from tqdm import tqdm
import numpy as np
import pytorch_ssim
from PIL import Image
from math import log10

import cv2
import math

path='../Desktop/City100/City100_NikonD5500/'
#ppath='../Desktop/City100/cv2test/'


bar = tqdm([1, 16, 48, 60, 98], desc='the bar')

only_y=False
for i in bar:
    lr_path = path + '%03dL.png' % i
    hr_path = path + '%03dH.png' % i

    lr=Image.open(lr_path)
    hr=Image.open(hr_path)

    lrr=cv2.imread(lr_path)
    hrr=cv2.imread(hr_path)



    lr=np.asfarray(lr)
    hr=np.asfarray(hr)

    lrr=np.asfarray(lrr)
    hrr=np.asfarray(hrr)


    print(lr[np.not_equal(lr,lrr)],lrr[np.not_equal(lr,lrr)])

    if only_y:
        rlt_h = np.dot(hr, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        rlt_l = np.dot(lr, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        #rlt_h = np.dot(hrr, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        #rlt_l = np.dot(lrr, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        #rlt_h = np.matmul(np.asfarray(hrr), [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
        #                      [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]

        #rlt_l = np.matmul(np.asfarray(lrr), [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                                            #                      [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]

        rlt_h = np.matmul(np.asfarray(hr), [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                            [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

        rlt_l = np.matmul(np.asfarray(lr), [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                            [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    rlt_h = rlt_h.round().astype('uint8')
    rlt_l=rlt_l.round().astype('uint8')


    mse=((np.asfarray(rlt_h) - np.asfarray(rlt_l)) ** 2).mean()


    psnr = 20 * log10(255 / np.sqrt(mse))
    print(psnr)


