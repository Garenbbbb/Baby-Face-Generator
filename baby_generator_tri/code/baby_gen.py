from face_match import face_landmark_finder
from tri import create_tri
from morph import morph_img_output

import sys
import pathlib
import argparse
import random
import cv2

# sys.path.insert(0, '/Users/changkehang/Desktop/MCS_Sem3/CS445/final project/Face-Morphing-git/code/CNN')
from cnn_ethnicity import predict 
from feathering import img_enhance

DIR_p = 'output/parent'
pathlib.Path(DIR_p).mkdir(parents=True, exist_ok=True)

DIR_m = 'output/mix'
pathlib.Path(DIR_m).mkdir(parents=True, exist_ok=True)

def startMorph(im1, im2, time_last, f, dir):

	[size, img1, img2, points1, points2, half_arr] = face_landmark_finder(im1, im2)
	tri = create_tri(size[1], size[0], half_arr)
	morph_img_output(time_last, f, img1, img2, points1, points2, tri, size, dir)

if __name__ == "__main__":

	# im1_file = "aligned_w.png"
	# im2_file = "aligned_m.png"

	parser = argparse.ArgumentParser(description='baby face generator')
	parser = argparse.ArgumentParser(description=
    'The baby face generator is \
    utilizing auto-alignment, triangulation morphing, CNN classifier')
	parser.add_argument('--man_img', type=str, required=True, help='need a pic of a man')
	parser.add_argument('--woman_img', type=str, required=True, help='need a pic of a woman')
	parser.add_argument('--region', type=str, help=
	'optional param to decide the average baby to morph; \
		choose from white, black, euro, asian')

	args = parser.parse_args()
	
	im1 = cv2.imread(args.man_img)
	im2 = cv2.imread(args.woman_img)
	time_last = 5
	f = 20

	startMorph(im1, im2, time_last, f, DIR_p)

	parent_mix_im = cv2.imread('output/parent/morph50.jpg')

	# race_arr_UTK = ["White", "Black", "Asian", "Indian", "Others"]
	if args.region is not None:
		if args.region == 'white':
			im3 = cv2.imread('data/white_b.png')
		elif args.region == 'black':
			im3 = cv2.imread('data/black_b.png')
		elif args.region == 'euro':
			im3 = cv2.imread('data/euro_b.png')
		elif args.region == 'asian':
			im3 = cv2.imread('data/asian_b.png')
	else:
			idx = predict(args.man_img)
			if idx == 0:
				rand_n = random.randint(0, 9)
				if rand_n % 2 == 1:
					im3 = cv2.imread('data/white_b.png')
				else:
					im3 = cv2.imread('data/euro_b.png')
			elif idx == 1:
				im3 = cv2.imread('data/black_b.png')
			elif idx == 2:
				im3 = cv2.imread('data/asian_b.png')
			else:
				rand_n = random.randint(0, 2)
				if rand_n == 0: im3 = cv2.imread('data/white_b.png')
				if rand_n == 1: im3 = cv2.imread('data/black_b.png')
				if rand_n == 2: im3 = cv2.imread('data/asian_b.png')

	startMorph(parent_mix_im, im3, time_last, f, DIR_m)
	baby_pic = cv2.imread('output/mix/morph50.jpg')
	cv2.imshow('baby image', baby_pic)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# img_enhance('output/mix/morph50.jpg')

	