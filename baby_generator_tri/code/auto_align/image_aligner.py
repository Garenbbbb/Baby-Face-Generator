from align_face import image_align, facemarks_finder



if __name__ == "__main__":
    """
    align img to be suitable for face morphing 
    """

    img_name = "unaligned.png" 
    output_s = 1024


    print('aligning {}'.format(img_name))
    print("starting.....")

    face_landmarks_set = facemarks_finder(img_name)
    for face_landmarks in face_landmarks_set:
        image_align(img_name, "output/aligned1.png", face_landmarks, output_size=output_s)

    
