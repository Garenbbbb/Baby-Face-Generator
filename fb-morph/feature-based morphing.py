import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw, ImageFilter
from helper import get_arguments, to_coordinates, get_landmark_lines_fixed, get_intermediate_face_outline, get_intermidate_lines, get_image_bokeh_effect, perpendicular_vector

## Part 0.1: read data
datadir = "D:/1 FA2022/0CS445/Final MP/fb-morph/"
gender = "GenderE"
image_dir_g = datadir + "images/" + gender
image_dir_c = datadir + "images/Children"
predictor_path = datadir + "pretrained/shape_predictor_68_face_landmarks.dat"
image_paths = os.listdir(image_dir_c)

## Part 0.2: set parameters
image_size = 256
b = 2
p = 0.5
a = 1
alpha = 0.5

feature_location = {
    "left_eye": np.arange(36, 42),
    "right_eye": np.arange(42, 48),
    "left_eyebrow": np.arange(17, 22),
    "right_eyebrow": np.arange(22, 27),
    "nose": np.arange(31, 36),
    "nose_bridge": np.arange(27, 31),
    "inner_lips": np.arange(60, 68),
    "outer_lips": np.arange(48, 60),
    "face": np.arange(0, 17),
}

# 2 utility functions
## 2.1 crop and resize
def feature_based_crop_and_resize(path, detector, predictor, size=256):
    """
    This function will crop then resize the input image and return the grey and color version
    Args:
        path: path of the image to be processed
        detector: dlib detector
        predictor: pretrained predictor
        size: image size. Defaults to 256.

    Returns:
        resized grey image and color image
    """
    image_gray = np.array(Image.open(path).convert("L"))
    image_color = np.array(Image.open(path).convert("RGB"))

    height, width = image_gray.shape[0], image_gray.shape[1]

    rect = detector(image_gray, 1)[0]
    input_image_coor = predictor(image_gray, rect)
    input_image_coor = to_coordinates(input_image_coor)
    
    # get border and crop
    padding = 10
    width_max, width_min = np.max(input_image_coor[:, 0]), np.min(input_image_coor[:, 0])
    height_max, height_min = np.max(input_image_coor[:, 1]), np.min(input_image_coor[:, 1])
    cropped_image_gray = image_gray[
        max(0, height_min - padding):min(height_max + padding, height), 
        max(0, width_min - padding):min(width_max + padding, width)
    ]
    cropped_image_color = image_color[
        max(0, height_min - padding):min(height_max + padding, height), 
        max(0, width_min - padding):min(width_max + padding, width)
    ]

    resized_image_gray = cv2.resize(cropped_image_gray, dsize=(size, size))
    resized_image_color = cv2.resize(cropped_image_color, dsize=(size, size))
    return resized_image_gray, resized_image_color

## 2.2 get line
def get_line_two_ends(img, detector, predictor):
    """
    This function output the start point and end point for each line
    Args:
        img: image to be processed
        detector: dlib detector
        predictor: pretrained predictor
    Returns:
        P, Q: start and end position of the lines
        lines: the lines in the picture
    """
    rect = detector(img, 1)[0]
    input_image_coor = predictor(img, rect)
    input_image_coor = to_coordinates(input_image_coor)
    
    lines = get_landmark_lines_fixed(feature_location)
    P = input_image_coor[lines[:, 0]]
    Q = input_image_coor[lines[:, 1]]
    return P.astype(np.float64), Q.astype(np.float64), input_image_coor, lines

## 2.3
def get_face_outline_coordinates(input_image_coor):
    facial_image_ids = [
        *feature_location["face"],
        *feature_location["right_eyebrow"][::-1],
        *feature_location["left_eyebrow"][::-1]
    ]
    return input_image_coor[facial_image_ids]

## 2.4
def draw_lines_image(img, lines_start, lines_end, input_image_coor, path):
    """
    draw lines on image
    Args:
        img: input image
        path: path to save to
    """
    img = Image.fromarray(img)
    draw_result = ImageDraw.Draw(img)
    # draw points
    for x, y in input_image_coor:
        r = 2
        draw_result.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 0, 0, 255))
    # draw feature lines
    for (x1, y1), (x2, y2) in zip(lines_start, lines_end):
        draw_result.line([(x1, y1), (x2, y2)], fill=(0, 0, 255, 255))
    img.save(path)
    return

## 2.5 make face mask
def get_face_mask(face_outline, image_size):
    face_x = list(face_outline[:, 0])
    face_y = list(face_outline[:, 1])
    mask = Image.new("RGB", (image_size, image_size))
    draw_result = ImageDraw.Draw(mask)
    draw_result.polygon(list(zip(face_x, face_y)), fill = (255, 255, 255))
    mask = mask.filter(ImageFilter.GaussianBlur(10)) # soften mask
    mask = np.array(mask) * 1.0 / 255
    return mask

## 2.6 merge images
def warp_and_merge(img_gray_1, P_1, Q_1, img_gray_2, P_2, Q_2, alpha, face_mask = None):
    #get and save intermidate results
    P_inter, Q_inter = get_intermidate_lines(P_1, Q_1, P_2, Q_2, alpha)
    warped_1 = warp_from_source(img_gray_1, P_1, Q_1, P_inter, Q_inter)
    warped_2 = warp_from_source(img_gray_2, P_2, Q_2, P_inter, Q_inter)
    name1 = "inter_1"
    name2 = "inter_2"
    Image.fromarray(warped_1).save("progress/" + name1 + gender + ".png")
    Image.fromarray(warped_2).save("progress/" + name2 + gender + ".png")

    merged = warped_1 * alpha + warped_2 * (1 - alpha)
    merged = merged.astype(np.uint8)

    if face_mask is not None:
        merged = get_image_bokeh_effect(merged, face_mask)
    return merged

### 2.6 helper2 
def warp_from_source(img_s, P_s, Q_s, P_d, Q_d):
    assert img_s.shape[0] == img_s.shape[1]
    eps = 1e-8

    perp_d = perpendicular_vector(Q_d - P_d)
    perp_s = perpendicular_vector(Q_s - P_s)
    dest_line_vec = Q_d - P_d
    source_line_vec = Q_s - P_s

    image_size = img_s.shape[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    X_d = np.dstack([x, y])
    X_d = X_d.reshape(-1, 1, 2)
    to_p_vec = X_d - P_d
    to_q_vec = X_d - Q_d
    u = np.sum(to_p_vec * dest_line_vec, axis=-1) / (np.sum(dest_line_vec**2, axis=1) + eps)
    v = np.sum(to_p_vec * perp_d, axis=-1) / (np.sqrt(np.sum(dest_line_vec**2, axis=1)) + eps)

    X_s = np.expand_dims(P_s, 0) + \
        np.expand_dims(u, -1) * np.expand_dims(source_line_vec, 0) + \
        np.expand_dims(v, -1) * np.expand_dims(perp_s, 0) / (np.sqrt(np.sum(source_line_vec**2, axis=1)).reshape(1, -1, 1) + eps)
    D = X_s - X_d
    to_p_mask = (u < 0).astype(np.float64)
    to_q_mask = (u > 1).astype(np.float64)
    to_line_mask = np.ones(to_p_mask.shape) - to_p_mask - to_q_mask

    to_p_dist = np.sqrt(np.sum(to_p_vec**2, axis=-1))
    to_q_dist = np.sqrt(np.sum(to_q_vec**2, axis=-1))
    to_line_dist = np.abs(v)
    dist = to_p_dist * to_p_mask + to_q_dist * to_q_mask + to_line_dist * to_line_mask
    dest_line_length = np.sqrt(np.sum(dest_line_vec**2, axis=-1))
    weight = (dest_line_length**p) / (((a + dist))**b + eps)
    weighted_D = np.sum(D * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + eps)

    X_d = X_d.squeeze()
    X_s = X_d + weighted_D
    X_s_ij = X_s[:, ::-1]

    if len(img_s.shape) == 2:
        warped = map_coordinates(img_s, X_s_ij.T, mode="nearest")
    else:
        warped = np.zeros((image_size*image_size, img_s.shape[2]))
        for i in range(img_s.shape[2]):
            warped[:, i] = map_coordinates(img_s[:, :, i], X_s_ij.T, mode="nearest")
    warped = warped.reshape(image_size, image_size, -1).squeeze()
    return warped.astype(np.uint8)

# 3 general functions
## 3.1 total morph
def morph_image(image_path_1, image_path_2, alpha, use_face_mask=False):
    """
    The function implement image morphing based on input images and user requirements
    Returns: the morphed image
    """    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    #resize: implemented at function 2.1
    img_gray_1, image_color_1 = feature_based_crop_and_resize(image_path_1, detector, predictor, image_size)
    img_gray_2, image_color_2 = feature_based_crop_and_resize(image_path_2, detector, predictor, image_size)
    
    #get line: implemented at function 2.2
    P_1, Q_1, input_image_coor_1, lines_1 = get_line_two_ends(img_gray_1, detector, predictor)
    P_2, Q_2, input_image_coor_2, lines_2  = get_line_two_ends(img_gray_2, detector, predictor)
    #you can print print("Number of lines:", len(lines_1))
   
    #get outline: implemented at function 2.3
    face1_outline = get_face_outline_coordinates(input_image_coor_1)
    face2_outline = get_face_outline_coordinates(input_image_coor_2)
    
    #draw lines to the landmark image: implemented at function 2.4
    name1 = "im1_line"
    name2 = "im2_line"
    draw_lines_image(image_color_1, P_1, Q_1, input_image_coor_1, "progress/" + name1 + gender+ ".jpg")
    draw_lines_image(image_color_2, P_2, Q_2, input_image_coor_2, "progress/" + name2 + gender+ ".jpg")
    
    #face_mask processing: implemented at function 2.5
    if use_face_mask:
        face_mask = get_face_mask(get_intermediate_face_outline(face1_outline, face2_outline, alpha), image_size)
    else:
        face_mask = None
        
    #merge two images: implemented at function 2.6
    merged = warp_and_merge(image_color_1, P_1, Q_1, image_color_2, P_2, Q_2, alpha, face_mask)
    return Image.fromarray(merged)

## 3.2 operation
def get_default_morph():
    """
    Activation function which automatically draw two pictures from the directory
    and apply the morph
    Returns: the morphed image by the two input images
    """
    # get image paths
    image_path_1 = os.path.join(image_dir_c, image_paths[0])
    image_path_2 = os.path.join(image_dir_c, image_paths[1])
    return morph_image(image_path_1, image_path_2, 0.5, True)

if __name__ == "__main__":
    args = get_arguments()

    if args.input1_path == "default":
        img = get_default_morph()
        img.save(datadir + "result/morph.jpg")
        print("done default sample morphing")
    else:
        a = args.a
        b = args.b
        p = args.p
        predictor_path = args.predictor_path
        image_size = args.image_size
        img = morph_image(args.input1_path, args.input2_path, args.alpha, args.bokeh)
        img.save(args.output)
        print("done assigned sample morphing")
    