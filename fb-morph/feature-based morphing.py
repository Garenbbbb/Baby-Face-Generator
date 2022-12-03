import os
import cv2
import dlib
import numpy as np
import time as time
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw, ImageFilter
from helper import get_arguments, get_coordinate, get_landmark_lines_fixed, get_intermediate_face_outline, get_intermidate_lines, get_image_bokeh_effect, get_vertical_vector

## Part 0.1: read data
datadir = "D:/1 FA2022/0CS445/Final MP/fb-morph/"
race = " Afr"
gender = False
image_dir_g = datadir + "images/GenderAfr"
image_dir_c = datadir + "images/ChildrenAfr"

predictor_path = datadir + "pretrained/shape_predictor_68_face_landmarks.dat"

if gender == True:
    type = " Gender"
    input_dir = image_dir_g
else:
    type = " Children"
    input_dir = image_dir_c
    
image_paths = os.listdir(input_dir)
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
    
    '''
    rect = detector(image_gray, 1)[0]
    input_image_coor = predictor(image_gray, rect)
    input_image_coor = to_coordinates(input_image_coor)
    '''
    input_image_coor = get_coordinate(image_gray, detector, predictor)
    
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
    '''
    rect = detector(img, 1)[0]
    input_image_coor = predictor(img, rect)
    input_image_coor = to_coordinates(input_image_coor)
    '''
    input_image_coor = get_coordinate(img, detector, predictor)
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
def get_mask_face(face_outline, image_size):
    face_x = list(face_outline[:, 0])
    face_y = list(face_outline[:, 1])
    mask = Image.new("RGB", (image_size, image_size))
    draw_result = ImageDraw.Draw(mask)
    draw_result.polygon(list(zip(face_x, face_y)), fill = (255, 255, 255))
    mask = mask.filter(ImageFilter.GaussianBlur(10)) # soften mask
    mask = np.array(mask) * 1.0 / 255
    return mask

## 2.6 merge images
def warp_then_merge(im1_gray, im1_P, im1_Q, im2_gray, im2_P, im2_Q, alpha, mask_face = None):
    #get and save intermidate results
    P_compo, Q_compo = get_intermidate_lines(im1_P, im1_Q, im2_P, im2_Q, alpha)
    warped_1 = warp_implementation(im1_gray, im1_P, im1_Q, P_compo, Q_compo)
    warped_2 = warp_implementation(im2_gray, im2_P, im2_Q, P_compo, Q_compo)
    name1 = "inter_1"
    name2 = "inter_2"
    Image.fromarray(warped_1).save("progress/" + name1 + race + type + ".png")
    Image.fromarray(warped_2).save("progress/" + name2 + race + type + ".png")

    merged = warped_1 * alpha + warped_2 * (1 - alpha)
    merged = merged.astype(np.uint8)

    if mask_face is not None:
        merged = get_image_bokeh_effect(merged, mask_face)
    return merged

### 2.6 helper3 
def warp_implementation(source_image, source_P, source_Q, destination_P, destination_Q):
    assert source_image.shape[0] == source_image.shape[1]
    terminate = 1e-8

    perdestination_P = get_vertical_vector(destination_Q - destination_P)
    persource_P = get_vertical_vector(source_Q - source_P)
    lines_destination = destination_Q - destination_P
    lines_source = source_Q - source_P

    image_size = source_image.shape[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    dim_hori = np.dstack([x, y])
    dim_hori = dim_hori.reshape(-1, 1, 2)
    vec_reach_P = dim_hori - destination_P
    vec_reach_Q = dim_hori - destination_Q
    u = np.sum(vec_reach_P * lines_destination, axis=-1) / (np.sum(lines_destination**2, axis=1) + terminate)
    v = np.sum(vec_reach_P * perdestination_P, axis=-1) / (np.sqrt(np.sum(lines_destination**2, axis=1)) + terminate)

    source_hori = np.expand_dims(source_P, 0) + \
        np.expand_dims(u, -1) * np.expand_dims(lines_source, 0) + \
        np.expand_dims(v, -1) * np.expand_dims(persource_P, 0) / (np.sqrt(np.sum(lines_source**2, axis=1)).reshape(1, -1, 1) + terminate)
    D = source_hori - dim_hori
    mask_reach_P = (u < 0).astype(np.float64)
    mask_reach_Q = (u > 1).astype(np.float64)
    mask_reach_line = np.ones(mask_reach_P.shape) - mask_reach_P - mask_reach_Q

    Pist_reach_destination = np.sqrt(np.sum(vec_reach_P**2, axis=-1))
    Qist_reach_destination = np.sqrt(np.sum(vec_reach_Q**2, axis=-1))
    to_line_dist = np.abs(v)
    dist = Pist_reach_destination * mask_reach_P + Qist_reach_destination * mask_reach_Q + to_line_dist * mask_reach_line
    
    length_destination_line = np.sqrt(np.sum(lines_destination**2, axis=-1))
    weight = (length_destination_line**p) / (((a + dist))**b + terminate)
    weighted_D = np.sum(D * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + terminate)

    dim_hori = dim_hori.squeeze()
    source_hori = dim_hori + weighted_D
    source_hori_elements = source_hori[:, ::-1]

    if len(source_image.shape) == 2:
        warped = map_coordinates(source_image, source_hori_elements.T, mode="nearest")
    else:
        warped = np.zeros((image_size*image_size, source_image.shape[2]))
        for i in range(source_image.shape[2]):
            warped[:, i] = map_coordinates(source_image[:, :, i], source_hori_elements.T, mode="nearest")
    warped = warped.reshape(image_size, image_size, -1).squeeze()
    return warped.astype(np.uint8)

# 3 general functions
## 3.1 total morph
def morph_image(image_path_1, image_path_2, alpha, use_mask_face=False):
    """
    The function implement image morphing based on input images and user requirements
    Returns: the morphed image
    """    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    #resize: implemented at function 2.1
    im1_gray, image_color_1 = feature_based_crop_and_resize(image_path_1, detector, predictor, image_size)
    im2_gray, image_color_2 = feature_based_crop_and_resize(image_path_2, detector, predictor, image_size)
    
    #get line: implemented at function 2.2
    im1_P, im1_Q, input_image_coor_1, _ = get_line_two_ends(im1_gray, detector, predictor)
    im2_P, im2_Q, input_image_coor_2, _  = get_line_two_ends(im2_gray, detector, predictor)
    #you can print print("Number of lines:", len(lines_1))
   
    #get outline: implemented at function 2.3
    face1_outline = get_face_outline_coordinates(input_image_coor_1)
    face2_outline = get_face_outline_coordinates(input_image_coor_2)
    
    #draw lines to the landmark image: implemented at function 2.4
    name1 = "im1_line"
    name2 = "im2_line"
    draw_lines_image(image_color_1, im1_P, im1_Q, input_image_coor_1, "progress/" + name1 + race + type + ".jpg")
    draw_lines_image(image_color_2, im2_P, im2_Q, input_image_coor_2, "progress/" + name2 + race + type + ".jpg")
    
    #mask_face processing: implemented at function 2.5
    if use_mask_face:
        mask_face = get_mask_face(get_intermediate_face_outline(face1_outline, face2_outline, alpha), image_size)
    else:
        mask_face = None
        
    #merge two images: implemented at function 2.6
    merged = warp_then_merge(image_color_1, im1_P, im1_Q, image_color_2, im2_P, im2_Q, alpha, mask_face)
    return Image.fromarray(merged)

## 3.2 operation
def get_default_morph():
    """
    Activation function which automatically draw two pictures from the directory
    and apply the morph
    Returns: the morphed image by the two input images
    """
    # get image paths
    image_path_1 = os.path.join(input_dir, image_paths[0])
    image_path_2 = os.path.join(input_dir, image_paths[1])
    return morph_image(image_path_1, image_path_2, 0.5, True)

if __name__ == "__main__":
    args = get_arguments()

    if args.input1_path == "default":
        img = get_default_morph()
        if input_dir == image_dir_c:
            name = "morph"
        else:
            name = "mix"
        img.save(datadir + "result/" + name + race + type + ".jpg")
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
    