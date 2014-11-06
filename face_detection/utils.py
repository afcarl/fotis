import os, errno

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
