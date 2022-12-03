from argparse import ArgumentParser
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def get_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("--input1_path", type=str, help="Path of first image file", default="default")
    parser.add_argument("--input2_path", type=str, help="Path of second image file", default="")
    parser.add_argument("--output", type=str, help="Path of output file", default="")
    
    parser.add_argument("--image_size", type=int, help="Size of cropped and resized face", default=256)
    parser.add_argument("--predictor_path", type=str, 
        help="Path of pretrained face feature detection model", default="D:/1 FA2022/0CS445/Final MP/fb-morph/pretrained/shape_predictor_68_face_landmarks.dat")
    
    parser.add_argument("--alpha", type=float, help="Alpha for merged image", default=0.5)
    parser.add_argument("--bokeh", action="store_true", help="Whether to use bokeh effect", default=False)
    parser.add_argument("--a", type=float, help="Parameter (a) in formula", default=1)
    parser.add_argument("--b", type=float, help="Parameter (b) in formula", default=2)
    parser.add_argument("--p", type=float, help="Parameter (p) in formula", default=0.5)

    args = parser.parse_args()
    return args

## 2.1 helper
def to_coordinates(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return np.array(coords)

def get_coordinate(img, detector, predictor):
    rect = detector(img, 1)[0]
    input_image_coor = predictor(img, rect)
    input_image_coor = to_coordinates(input_image_coor)
    return input_image_coor
    
### 2.2 helper
def get_landmark_lines_fixed(feature_location):
    lines = []
    for key, pts in feature_location.items():
        coords = np.stack([pts[:-1], pts[1:]]).T
        lines.append(coords)
    lines = np.concatenate(lines)
    return lines


### 2.5 helper
def get_intermediate_face_outline(face1, face2, alpha = 0.5):
    return face1 * alpha + face2 * (1 - alpha)

### 2.6 helper 1
def get_intermidate_lines(im1_P, im1_Q, im2_P, im2_Q, alpha = 0.5):
    """
    alpha blending lines of images
    """
    P = im1_P * alpha + im2_P * (1 - alpha)
    Q = im1_Q * alpha + im2_Q * (1 - alpha)
    return P, Q
### 2.6 helper 2 
def get_image_bokeh_effect(face, mask):
    blurred = Image.fromarray(face).filter(ImageFilter.GaussianBlur(2))
    
    enhancer = ImageEnhance.Brightness(blurred)
    blurred = enhancer.enhance(0.6)
    
    blurred = np.array(blurred)
    res = face * mask + blurred * (1 - mask)
    res = res.astype(np.uint8)
    return res

#### 2.6 helper3 helper
def get_vertical_vector(v):
    v_length = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v_homo = np.pad(v, ((0, 0), (0, 1)), mode="constant") # pad to R3, pad zeros
    z_axis = np.zeros(v_homo.shape)
    z_axis[:, -1] = 1
    p = np.cross(v_homo, z_axis)
    p = p[:, :-1] # ignore z axis
    p_length = np.sqrt(np.sum(p**2, axis=1, keepdims=True))
    p = p / (p_length + 1e-8) # now sum = 1
    p *= v_length
    return p