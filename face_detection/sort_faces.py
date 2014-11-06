# -*- coding: utf8 -*-
import csv, sys, os, cPickle, math, csv, shutil
from os.path import join
import cv2, numpy
import utils


def sort_faces(path_to_csv, path_to_faces, path_to_results):
	csv_file = open(path_to_csv, 'rb')
	try:
		reader = csv.reader(csv_file, delimiter=';')
		# the first line contains the name of the columns, so we pop it out
		reader.next()
		utils.mkdir(path_to_results)
		for row in reader:
			image_file_name = os.path.basename(row[0])
			person_name = row[1]
			files_names_list = [filename for filename in os.listdir(path_to_faces) if filename.startswith(os.path.splitext(image_file_name)[0])]
			# if no faces exist for this person, skip
			if not files_names_list:
				continue
			path_to_person_folder = join(path_to_results, person_name)
			# mkdir checks if exists and if not then creates
			utils.mkdir(path_to_person_folder)
			for face_file_name in files_names_list:
				one_face_path = join(path_to_faces, face_file_name)
				shutil.copy2(one_face_path, join(path_to_person_folder, face_file_name))
	finally:
		csv_file.close()

"""
sort_faces <CSV file> <faces folder> <results folder> 
"""
if (len(sys.argv) > 3):
    path_to_CSV_file = sys.argv[1]
    path_to_faces = sys.argv[2]
    path_to_results = sys.argv[3]
    sort_faces(path_to_CSV_file, path_to_faces, path_to_results)
else:
    raise KeyError('Not enough arguments')
