import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def img_enhance(test_file_name, gamma = 1.01):

    # read image
    test_img = cv2.imread(test_file_name)

    # # method#1 
    test_img_color = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img_gamma = np.power(test_img_color, gamma).clip(0,255).astype("uint8")

    # method#2 
    # source: https://stackoverflow.com/questions/61695773/how-to-set-the-best-value-for-gamma-correction
    # convert img to gray
    # test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # # compute gamma = log(mid*255)/log(mean)
    # mid = 0.5
    # mean = np.mean(test_img_gray)
    # gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)

    # # do gamma correction
    # test_img_gamma = np.power(test_img, gamma).clip(0,255).astype("uint8")
    


    ## citing: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
    # img_yuv = cv2.cvtColor(test_img_gamma, cv2.COLOR_BGR2YUV)

    # # equalize the histogram of the Y channel
    # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # # convert the YUV image back to RGB format
    # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # RGB_img = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    RGB_img = cv2.cvtColor(test_img_gamma, cv2.COLOR_BGR2RGB)
    cv2.imshow('enhanced baby image', RGB_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def histogram_qualization(image):
#     dst = cv2.equalizeHist(image)
#     # plt.imshow(dst,cmap='gray')
#     # plt.savefig("src.png")