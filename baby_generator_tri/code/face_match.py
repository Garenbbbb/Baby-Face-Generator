import numpy as np
import cv2
import dlib

LAND_MARK_COUNT = 68

def margin_helper(img_1, img_2):
  (h1, w1, _) = img_1.shape  # (h,w)
  (h2, w2, _) = img_2.shape
  diff_h = abs(h1-h2)//2
  diff_w = abs(w1-w2)//2
  avg_h = (h1+h2)//2
  avg_w = (w1+w2)//2
  return [(h1,w1), (h2,w2), diff_h, diff_w, avg_h, avg_w]

def crop_helpper(img_1,img_2):
    [(h1,w1), (h2,w2), diff_h, diff_w, avg_h, avg_w] = margin_helper(img_1,img_2)
   
    if(h1 == h2 and w1 == w2):
        return [img_1,img_2]

    elif(h1 <= h2 and w1 <= w2):
        return [img_1,img_2[-diff_h:avg_h,-diff_w:avg_w]]

    elif(h1 >= h2 and w1 >= w2):
        return [img_1[diff_h:avg_h,diff_w:avg_w],img_2]

    elif(h1 >= h2 and w1 <= w2):
        return [img_1[diff_h:avg_h,:],img_2[:,-diff_w:avg_w]]

    else:
        return [img_1[:,diff_w:avg_w],img_2[diff_h:avg_h,:]]

def cropper(img_1,img_2):
    [(h1,w1), (h2,w2), diff_h, diff_w, avg_h, avg_w] = margin_helper(img_1,img_2)

    if(h1 == h2 and w1 == w2):
        return [img_1,img_2]

    elif(h1 <= h2 and w1 <= w2):
        scale0 = h1/h2
        scale1 = w1/w2
        if(scale0 > scale1):
            res = cv2.resize(img_2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img_2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_helpper(img_1,res)

    elif(h1 >= h2 and w1 >= w2):
        scale0 = h2/h1
        scale1 = w2/w1
        if(scale0 > scale1):
            res = cv2.resize(img_1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img_1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_helpper(res,img_2)

    elif(h1 >= h2 and w1 <= w2):
        return [img_1[diff_h:avg_h,:],img_2[:,-diff_w:avg_w]]
    
    else:
        return [img_1[:,diff_w:avg_w],img_2[-diff_h:avg_h,:]] 




def face_landmark_finder(original_im1, original_im2):
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
    land_marks = np.zeros((LAND_MARK_COUNT,2))

    img_pair = cropper(original_im1, original_im2)
    list_img1 = []
    list_img2 = []
    flag = 0 

    for img in img_pair:
        (h,w,_) = img.shape
        if flag == 0:
            curr_list = list_img1
        else:
            curr_list = list_img2

        bbox_list = face_detector(img, 1)
        flag += 1

        for rect_bbox in bbox_list:
            shape_f = face_predictor(img, rect_bbox)
            for i in range(LAND_MARK_COUNT):
                x = shape_f.part(i).x
                y = shape_f.part(i).y
                curr_list.append((x, y))
                land_marks[i][0] += x
                land_marks[i][1] += y  
            curr_list.append((1,1))
            curr_list.append((w-1,1))
            curr_list.append(((w-1)//2,1))
            curr_list.append((1,h-1))
            curr_list.append((1,(h-1)//2))
            curr_list.append(((w-1)//2,h-1))
            curr_list.append((w-1,h-1))
            curr_list.append(((w-1),(h-1)//2))

    (h,w,_) = img_pair[1].shape
    half_arr = land_marks/2
    half_arr = np.append(half_arr,[[1,1]],axis=0)
    half_arr = np.append(half_arr,[[w-1,1]],axis=0)
    half_arr = np.append(half_arr,[[(w-1)//2,1]],axis=0)
    half_arr = np.append(half_arr,[[1,h-1]],axis=0)
    half_arr = np.append(half_arr,[[1,(h-1)//2]],axis=0)
    half_arr = np.append(half_arr,[[(w-1)//2,h-1]],axis=0)
    half_arr = np.append(half_arr,[[w-1,h-1]],axis=0)
    half_arr = np.append(half_arr,[[(w-1),(h-1)//2]],axis=0)
    
    return [(h,w),img_pair[0],img_pair[1],list_img1,list_img2,half_arr]


