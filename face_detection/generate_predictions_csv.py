import csv, sys, os, cPickle
from os.path import join
import numpy
from utils import mkdir

# ###################################
#          START CONSTANTS
# ###################################
DEFAULT_THRESHOLD = 0.99

# #################################
# 	    END CONSTANTS
# #################################

def unpickle(path_to_file):
    open_file = open(path_to_file, 'rb')  # read binary
    data_dict = cPickle.load(open_file)
    open_file.close()
    return data_dict

def generate_predictions_csv(path_to_batch, path_to_predictions, path_to_meta, path_to_csv, threshold=DEFAULT_THRESHOLD):
    batch = unpickle(path_to_batch)
    predictions = unpickle(path_to_predictions)
    meta = unpickle(path_to_meta)
    path, filename = os.path.split(path_to_csv)
    if path:
        mkdir(path)
    # predicted persons
    preds = numpy.argmax(predictions['data'], axis=1)
    # the estimation percentage of predition being right
    percentages = numpy.amax(predictions['data'], axis=1)
    # truth table which tells if the prediction was correct
    results = predictions['labels'] == preds

    csv_file = open(path_to_csv, 'wb')
    try:
        writer = csv.writer(csv_file,  delimiter=';',)
        writer.writerow( ('Filename', 'Person', 'Percentage', 'Correct') )
        for i in range(len(results[0])):
            if percentages[i] < threshold:
                continue
            writer.writerow( (batch["filenames"][i], meta["label_names"][preds[i]], "%.2f" % (percentages[i] * 100), results[0][i]) )
    finally:
        csv_file.close()


"""
input:
batch_file, preds_file, batch.meta, csv_file, threshold
output:
csv: : file_name, person, percentage 
"""
if (len(sys.argv) > 4):
        batch_path = sys.argv[1]
        predictions_path = sys.argv[2]
        meta_path = sys.argv[3]
        csv_path = sys.argv[4]
        threshold = DEFAULT_THRESHOLD
        if (len(sys.argv) > 5):
            threshold = float(sys.argv[5])
	generate_predictions_csv(batch_path, predictions_path, meta_path, csv_path, threshold)
else:
	raise KeyError('Not enough arguments')
