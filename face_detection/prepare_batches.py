# -*- coding: utf8 -*-
import csv, sys, os, cPickle, math, random, unicodedata
from os.path import join
import cv2, numpy
import utils

DEFAULT_FACE_SIZE = 32

def prepare_image(image, grayscale=False):
    if grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prepared_image = numpy.zeros((gray.size), dtype="uint8")
        bgr_index = 0
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                prepared_image[bgr_index] = gray[i, j]
                bgr_index+=1
        return prepared_image
    else:
        green_start = image.size / 3  # 1/3 of batched_image size
        blue_start = green_start * 2  # 2/3 of batched_image size
        # red_start = 0
        prepared_image = numpy.zeros((image.size), dtype="uint8")
        bgr_index = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                prepared_image[bgr_index + blue_start] = image[i, j, 0]
                prepared_image[bgr_index + green_start] = image[i, j, 1]
                prepared_image[bgr_index] = image[i, j, 2]
                bgr_index+=1
        return prepared_image

def restore_image(batched_image):
    green_start = batched_image.size / 3  # 1/3 of batched_image size
    blue_start = green_start * 2  # 2/3 of batched_image size
    # red_start = 0
    batched_image_size = int(math.sqrt(green_start))  # sqrt of batched_image/3, the length of the edge
    restored_image = numpy.zeros((batched_image_size, batched_image_size, 3), dtype="uint8")
    bgr_index = 0
    for i in range(batched_image_size):
        for j in range(batched_image_size):
            restored_image[i,j]=[batched_image[bgr_index + blue_start],
                                 batched_image[bgr_index + green_start],
                                 batched_image[bgr_index]
                                 ]
            bgr_index+=1
    return restored_image


def pickle(data, filename):
    output = open(filename, 'wb')  # write binary
    # a third argument may be given - protocol; -1 is the highest
    # cPickle.dump(data, output, -1)
    cPickle.dump(data, output)
    output.close()


def unpickle(path_to_file):
    open_file = open(path_to_file, 'rb')  # read binary
    data_dict = cPickle.load(open_file)
    open_file.close()
    return data_dict


def folder_tree_to_dictionary(path_to_folder):
    """
    Assumes that root is folder that has a number of folders in it. Those folders have
    files in them
    """
    folder_tree_dictionary = {}
    folders_list = os.listdir(path_to_folder)
    for folder in folders_list:
        folder_tree_dictionary[folder] = os.listdir(join(path_to_folder, folder))
    return folder_tree_dictionary


def level_list(alist, desired_length):
    """
    Levels list to desired length.
    """
    # remove random elements if longer
    while len(alist) > desired_length:
        alist.pop(random.randrange(len(alist)))


def normalize_string(string):
    try:
        normalized = unicodedata.normalize('NFKD', string.decode("utf-8")).encode('ascii','ignore')
    except UnicodeDecodeError:
        normalized = unicodedata.normalize('NFKD', string.decode("latin-1")).encode('ascii','ignore')
    return normalized

"""
batch dictionary keys:
    data <type 'numpy.ndarray'> - transformed numpy images in numpy list
    labels <type 'list'> - nth index in meta[label_names]
    batch_label <type 'str'> batch name i of j a.la training batch 1 of 5
    filenames <type 'list'> nth in filenames is the image filename of nth in data

meta dictionary keys:
    num_cases_per_batch <type 'int'> - number of samples in batch
    label_names <type 'list'> - list of person names
    num_vis <type 'int'> - image.size
"""
def create_batches(path_to_structured_folder_tree, path_to_results, num_cases_per_batch, min_images, max_images, image_size=DEFAULT_FACE_SIZE, use_grayscale=False):
    # PART 0 - preparations
    utils.mkdir(path_to_results)
    meta = {"label_names": [],
            "num_cases_per_batch": num_cases_per_batch,
            "img_size": image_size,
            "num_vis": image_size * image_size }  # being used below
    if not use_grayscale:
        meta["num_vis"] *= 3

    # PART 1 - scan directories
    print "Scanning directories..."
    folder_tree_dictionary = folder_tree_to_dictionary(path_to_structured_folder_tree)
    person_indexes = {}
    for folder in folder_tree_dictionary.keys():
        if (len(folder_tree_dictionary[folder]) >= min_images):
            level_list(folder_tree_dictionary[folder], max_images)
            # create dictionary of person names and their respective index in meta
            person_indexes[folder] = len(meta["label_names"])
            meta["label_names"].append(normalize_string(folder))
        else:
            del folder_tree_dictionary[folder]

    total_nr_of_images = sum([len(folder_tree_dictionary[folder]) for folder in folder_tree_dictionary])
    print "Total number of images: ", total_nr_of_images
    print "Number of classes: ", len(meta["label_names"])

    # PART 2 - load images
    print "Loading images..."
    data = numpy.empty((meta["num_vis"], total_nr_of_images),  dtype="uint8")
    data_index = 0
    labels = []
    filenames = []
    for folder in folder_tree_dictionary:
        for file_name in folder_tree_dictionary[folder]:
            file_path = join(path_to_structured_folder_tree, folder, file_name)
            face = cv2.imread(file_path)
            face = cv2.resize(face, (image_size, image_size))

            data[:, data_index] = prepare_image(face, use_grayscale)
            labels.append(person_indexes[folder])
            filenames.append(os.path.basename(file_name))
            data_index += 1

    # PART 3 - dump metadata
    print "Writing batches.meta"
    # calculate mean of the images
    meta['data_mean'] = numpy.mean(data, 1)
    pickle(meta, join(path_to_results, "batches.meta"))

    # PART 4 - create batches
    random_indexes = range(total_nr_of_images)
    random.shuffle(random_indexes)
    nr_of_batches = int(math.ceil(total_nr_of_images / float(num_cases_per_batch)))
    for i in range(nr_of_batches):
        batch_start = i * num_cases_per_batch
        batch_end = min((i + 1) * num_cases_per_batch, total_nr_of_images)
        batch_range = random_indexes[batch_start:batch_end]
        batch = {"data": data[:, batch_range],
                 "labels": [labels[j] for j in batch_range],
                 "filenames": [filenames[j] for j in batch_range],
                 "batch_label": "data batch %d of %d" % (i, nr_of_batches)}

        print "Writing batch nr %d with %d images" % (i, batch_end - batch_start)
        pickle(batch, join(path_to_results, "data_batch_%d" % i))

"""
prepare_batches.py <folder with people-named-folders> <min nr of images>
"""
if (len(sys.argv) > 4):
    path_to_structured_folder_tree = sys.argv[1]
    path_to_results = sys.argv[2]
    num_cases_per_batch = int(sys.argv[3])
    min_images = 0
    if (len(sys.argv) > 4):
        min_images = int(sys.argv[4])
    max_images = 1000
    if (len(sys.argv) > 5):
        max_images = int(sys.argv[5])
    face_size = DEFAULT_FACE_SIZE
    if (len(sys.argv) > 6):
        face_size = int(sys.argv[6])
    use_gray = False
    if (len(sys.argv) > 7) and (sys.argv[7].lower() == "true"):
        use_gray = True
    create_batches(path_to_structured_folder_tree, path_to_results, num_cases_per_batch, min_images, max_images, face_size, use_gray)
else:
    raise KeyError('Not enough arguments')
