import numpy as np
import PIL.Image
from img_tool import img_process
import dlib

def image_align(src, dst, landmarks, output_size=1024):
    
    features = np.array(landmarks)
    # feature vector
    left_top  = np.mean(features[36 : 42], axis = 0)
    right_top = np.mean(features[42 : 48], axis = 0)
    
    top_avg = (left_top + right_top) / 2.0
    top_dif = right_top - left_top

    bot_left = features[48]
    bot_right = features[54]

    bot_avg = (bot_left + bot_right) / 2.0
    top_bot =  bot_avg - top_avg

    # find correct window
    k = 0.1
    p1 = 2.0
    p2 = 1.8

    var = top_dif - [-1, 1] * np.flipud(top_bot) 
    var = var / np.hypot(*var)
    var = var * max(np.hypot(*top_dif) * p1, np.hypot(*top_bot) * p2)
    r = [-1, 1] * np.flipud(var) 

    center = top_avg + top_bot * k

    l1 = center - var - r
    l2 =  center - var + r
    l3 = center + var + r
    l4 = center + var - r

    b_box = np.stack([l1, l2, l3, l4])
    diag_size = np.hypot(*var) * 2

    img = PIL.Image.open(src).convert('RGBA').convert('RGB')
    img = img_process(img, b_box, diag_size)

    # Save aligned image.
    img.save(dst, 'PNG')


def facemarks_finder(img):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    img = dlib.load_rgb_image(img)
    dets = detector(img, 1)

    face_landmarks_sets = []
    for detection in dets:
         # yield face_landmarks  # yield saves memory
        face_landmarks = [(item.x, item.y) for item in shape_predictor(img, detection).parts()]
        face_landmarks_sets.append(face_landmarks)
    return face_landmarks_sets