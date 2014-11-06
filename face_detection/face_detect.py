# -*- coding: utf8 -*-
import os, sys, errno, math
from os.path import join
from time import strftime, gmtime, time
import numpy as np
import cv2
import utils

# ###################################
# 			START CONSTANTS
# ###################################
DEBUG_MODE = True

FACE_CASCADE_CLASSIFIERS = {
	"alt": "haarcascade_frontalface_alt.xml",				# 0
	"alt2": "haarcascade_frontalface_alt2.xml",				# 1
	"alt_tree": "haarcascade_frontalface_alt_tree.xml",		# 2
	"default": "haarcascade_frontalface_default.xml", 		# 3
	"profileface": "haarcascade_profileface.xml"			# 4
}
FACE_CASCADE_CLASSIFIER = join('cascades', FACE_CASCADE_CLASSIFIERS["default"])
# the number used to multiply the box surrounding the face
FACE_RECTANGLE_MULTIPLIER = 1.4
# the size the face images are resized into
DEFAULT_MIN_FACE_SIZE = 28

SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 5
# Flags:
# CV_HAAR_DO_CANNY_PRUNING
# CV_HAAR_SCALE_IMAGE  # fastest
# CV_HAAR_FIND_BIGGEST_OBJECT
# CV_HAAR_DO_ROUGH_SEARCH
FLAG = cv2.cv.CV_HAAR_SCALE_IMAGE
# #################################
# 			END CONSTANTS
# #################################

def debugger(text, newline=True):
	if DEBUG_MODE:
		if newline:
			print text
		else:
			print text,

def enlarge_rectangle(x, y, w, h, factor = FACE_RECTANGLE_MULTIPLIER):
	new_w = int((w * factor) + 0.5)
	new_h = int((h * factor) + 0.5)
	center_x = int(x + w / 2)
	center_y = int(y + h / 2)
	new_x = center_x - (new_w / 2)
	if new_x < 0:
		new_x = 0
	new_y = center_y - (new_h / 2)
	if new_y < 0:
		new_y = 0
	return new_x, new_y, new_w, new_h

def process_and_write_exhausting(path_to_folder, path_to_results, min_face_size=32):
	face_cascader = cv2.CascadeClassifier(FACE_CASCADE_CLASSIFIER)

	# results_root_path = join(path_to_results, strftime("%d-%m-%Y %H-%M-%S", gmtime()))
	utils.mkdir(path_to_results)

	nr_detected=0
	nr_not_detected=0
	t1 = time()
	for filename in os.listdir(path_to_folder):
		img = cv2.imread(join(path_to_folder, filename))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for sf in (1.45, 1.4, 1.35, 1.3, 1.25, 1.2):
			faces = face_cascader.detectMultiScale(
					gray,
					scaleFactor=sf,
					minNeighbors=MIN_NEIGHBORS,
					minSize=(min_face_size, min_face_size),
					flags=FLAG
					)
			found_faces = (not type(faces)==tuple) and faces.any()
			if found_faces:
				# the largest face is the last face, so we start numbering from the number of
				# faces since we start iterating from the first face which is the smallest
				face_number = faces.size / 4
				break

		for (x,y,w,h) in faces:
			face_number -= 1
			nx, ny, nw, nh = enlarge_rectangle(x, y, w, h, FACE_RECTANGLE_MULTIPLIER)
			face = img[ny:ny+nh, nx:nx+nw]
			if not (face.shape[0] == face.shape[1]):
				face = img[y:y+h, x:x+w]
			file_name, file_ext = os.path.splitext(filename)
			face_filename = file_name + "_face%d" % face_number + file_ext
			cv2.imwrite(join(path_to_results, face_filename), face)

		if found_faces:
			nr_detected += 1
		else:
			nr_not_detected += 1

		debugger("%s: %d faces" % (filename, faces.size / 4 if found_faces else 0));

	t2 = time()
	debugger("\timages with faces: %d" % nr_detected)
	debugger("\timages w/ou faces: %d" % nr_not_detected)
	debugger("\ttime taken: %.2fs" % (t2-t1))

"""
face_detect.py <piltide kataloog> <nägude kataloog> <näofaili suurus pikslites (ruudukujuline)>
Nt face_detect.py /storage/fotis/pildid /storage/fotis/naod 32
"""
if (len(sys.argv) > 2):
	path_to_source = sys.argv[1]
	path_to_results = sys.argv[2]
	min_face_size = DEFAULT_MIN_FACE_SIZE
	if len(sys.argv) > 3:
		min_face_size = int(sys.argv[3])
	process_and_write_exhausting(path_to_source, path_to_results, min_face_size)
else:
	raise KeyError('Not enough arguments')
