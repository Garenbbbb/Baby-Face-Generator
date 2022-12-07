import numpy as np
import cv2
from PIL import Image
import skimage


def affine_transform(src, tri1, tri2, size) :
    affine = cv2.getAffineTransform(tri1, tri2)
    return cv2.warpAffine(src, affine, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def tri_m(img1, img2, img, k1, k2, k, alpha) :

    # Find bounding rectangle for each triangle
    l1 = cv2.boundingRect(np.float32([k1]))
    l2 = cv2.boundingRect(np.float32([k2]))
    l = cv2.boundingRect(np.float32([k]))

    area = np.zeros((l[3], l[2], 3), dtype = np.float32)

    w1 = []
    w2 = []
    w = []

    for i in range(3):

        w1.append(((k1[i][0] - l1[0]),(k1[i][1] - l1[1])))
        w.append(((k[i][0] - l[0]),(k[i][1] - l[1])))
        w2.append(((k2[i][0] - l2[0]),(k2[i][1] - l2[1])))

    p = (1.0, 1.0, 1.0)
    s = 16

    cv2.fillConvexPoly(area, np.int32(w), p, s, 0)


    # Apply warpImage to small rectangular patches
    im1r = img1[l1[1]:l1[1] + l1[3], l1[0]:l1[0] + l1[2]]
    im2r = img2[l2[1]:l2[1] + l2[3], l2[0]:l2[0] + l2[2]]

    size = (l[2], l[3])

    w = np.float32(w)
    w1 =  np.float32(w1)
    w2 =  np.float32(w2)

    res1 = affine_transform(im1r, w1, w, size)
    res2 = affine_transform(im2r, w2, w, size)


    iout = (1.0 - alpha) * res1 + alpha * res2
    new_patch = img[l[1]:l[1]+l[3], l[0]:l[0]+l[2]] * ( 1 - area ) + iout * area
    img[l[1]:l[1]+l[3], l[0]:l[0]+l[2]] = new_patch

def morph_img_output(time_last, f, img1, img2, points1, points2, tri_list, size, dir):
    for j in range(int(time_last * f)):
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        cur = []
        result = np.zeros(img1.shape, dtype = img1.dtype)
        factor = j / (int(time_last * f) - 1)

        for i in range(len(points1)):
            a = (1 - factor) * points1[i][0] + factor * points2[i][0]
            b = (1 -  factor) * points1[i][1] + factor * points2[i][1]
            cur.append((a,b))

        for i in range(len(tri_list)):
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [cur[x], cur[y], cur[z]]

            tri_m(img1, img2, result, t1, t2, t,  factor)


        res = Image.fromarray(cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB))
        skimage.io.imsave(dir + "/morph{}.jpg".format(j), np.array(res))
