import sys
import os
import numpy as np
import shutil

###############################################################################
# RUN 1 TIMES TO SPLIT DATA INTO TRAIN AND TEST
###############################################################################


def get_all_parts(data_folder):
    training_data = "training_data"
    files = os.listdir(os.path.join(data_folder, training_data))
    parts = list()
    for file in files:
        if file.endswith(".txt"):
            num = int(os.path.splitext(file)[0])
            parts.append(num)
    return parts


def get_split_data(parts, train_percentage):
    """
    Split data by name of parts.

    Args:
        parts (list): all parts which get by name of text file
        train_percentage (int): percentage of train data

    Returns:
        tuple: train data and test data
    """
    len_parts = len(parts)
    len_train = int(len_parts * train_percentage)
    # turn upside down parts  with numpy shuffle
    np.random.shuffle(parts)
    train_set = parts[:len_train]
    test_set = parts[len_train:]
    return train_set, test_set


def split_data(data_folder, train_percentage):
    """
    Split data by name of parts.
    Args:
        data_folder (str): folder of data
        train_percentage (int): percentage of train data
    Returns:
        tuple: train data and test data
    """
    parts = get_all_parts(data_folder)
    test_folder = "test_data"
    train_folder = "training_data"
    files = os.listdir(os.path.join(data_folder, train_folder))
    os.makedirs(os.path.join(data_folder, test_folder), exist_ok=True)
    _, test_set = get_split_data(parts, train_percentage)
    for part in test_set:
        for f in files:
            if f.startswith(str(part)):
                shutil.move(os.path.join(data_folder, train_folder, f),
                            os.path.join(data_folder, test_folder, f))
    print("Done")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("Too many arguments")
    data_folder = sys.argv[1]
    split_data(data_folder, 0.8)
