import numpy as np
import scipy.ndimage
import PIL.Image

def img_process(img, b_box, diag_size, output_s=1024, new_size=4096):
    # resize 
    resize_factor = int(np.floor(diag_size / output_s * 0.5))
    if resize_factor > 1:
        h = int(np.rint(float(img.size[0]) / resize_factor))
        w = int(np.rint(float(img.size[1]) / resize_factor))
        img = img.resize((h,w), PIL.Image.ANTIALIAS)
        b_box /= resize_factor
        diag_size /= resize_factor

    # cropping 
    border = max(int(np.rint(diag_size * 0.1)), 3)
    crop = []
    crop.append(int(np.floor(min(b_box[:,0]))))
    crop.append(int(np.floor(min(b_box[:,1]))))
    crop.append(int(np.ceil(max(b_box[:,0]))))
    crop.append(int(np.ceil(max(b_box[:,1]))))
    crop_update = []
    crop_update.append(max(crop[0] - border, 0))
    crop_update.append(max(crop[1] - border, 0))
    crop_update.append(min(crop[2] + border, img.size[0]))
    crop_update.append(min(crop[3] + border, img.size[1]))

    if crop_update[2] - crop_update[0] < img.size[0] or crop_update[3] - crop_update[1] < img.size[1]:
        img = img.crop(crop_update)
        b_box -= crop_update[0:2]

    # padding with mask 
    pad = []
    pad.append(int(np.floor(min(b_box[:,0]))))
    pad.append(int(np.floor(min(b_box[:,1]))))
    pad.append(int(np.ceil(max(b_box[:,0]))))
    pad.append(int(np.ceil(max(b_box[:,1]))))
    pad_update = []
    pad_update.append(max(-pad[0] + border, 0))
    pad_update.append(max(-pad[1] + border, 0))
    pad_update.append(max(pad[2] - img.size[0] + border, 0))
    pad_update.append(max(pad[3] - img.size[1] + border, 0))

    if max(pad_update)+4 > border:
        pad_factor = 0.3 
        pad_update = np.maximum(pad_update, int(np.rint(diag_size * pad_factor)))
        pad_mask = ((pad_update[1], pad_update[3]), (pad_update[0], pad_update[2]), (0, 0))
        img = np.pad(np.float32(img), pad_mask, 'reflect')

        h_img, w_img, _ = img.shape
        factor_y, factor_x, _ = np.ogrid[:h_img, :w_img, :1]

        h_min = np.minimum(np.float32(factor_x) / pad_update[0], np.float32(w_img-1-factor_x) / pad_update[2])
        w_min = np.minimum(np.float32(factor_y) / pad_update[1], np.float32(h_img-1-factor_y) / pad_update[3])
        mask = np.maximum(1.0 - h_min, 1.0 - w_min)

        b_factor = diag_size * 0.02
        mask_clip = np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        mask_median_clip = np.clip(mask, 0.0, 1.0)
        img += (scipy.ndimage.gaussian_filter(img, [b_factor, b_factor, 0]) - img) * mask_clip
        img += (np.median(img, axis=(0,1)) - img) * mask_median_clip
        img = np.uint8(np.clip(np.rint(img), 0, 255))

        img = PIL.Image.fromarray(img, 'RGB')
        b_box += pad_update[:2]

    # enlarge 
    img = img.transform((new_size, new_size), PIL.Image.QUAD, (b_box + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_s < new_size:
        img = img.resize((output_s, output_s), PIL.Image.ANTIALIAS)

    return img 