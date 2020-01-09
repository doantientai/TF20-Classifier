"""
    Given a directory of images
    Load model and run inference
    Return the raw prediction
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

from train import classifier, IMG_HEIGHT, IMG_WIDTH, list_classes

import os
from os.path import join
from os import makedirs
import numpy as np
from shutil import copyfile

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
AUTOTUNE = tf.data.experimental.AUTOTUNE

DIR_REAL_DATA = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA_1k'

# DIR_TEST = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_10k_generated'
# PATH_WEIGHTS = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_002_A_1k_baseline/models/weights.07-0.9443.hdf5'

# DIR_TEST = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_40k_generated'
# PATH_WEIGHTS = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_002_A_1k_baseline/models/weights.07-0.9443.hdf5'

# DIR_TEST = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_max_generated'
# PATH_WEIGHTS = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_002_A_1k_baseline/models/weights.07-0.9443.hdf5'

DIR_TEST = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_maxX3_generated'
PATH_WEIGHTS = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_002_A_1k_baseline/models/weights.07-0.9443.hdf5'

if DIR_TEST.endswith('/'):
    path_data_inference = DIR_TEST[:-1]
path_out_root = DIR_TEST + '_split'
path_out_text = join(path_out_root, 'split.txt')
path_out_keep = join(path_out_root, "keep")
path_out_drop = join(path_out_root, "drop")


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == list_classes


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label, file_path


def test_model(model):
    list_file_paths = []
    for path, subdirs, files in os.walk(DIR_TEST):
        for name in files:
            list_file_paths.append(os.path.join(path, name))
        for subdir in subdirs:
            makedirs(join(path_out_keep, subdir), exist_ok=True)
            makedirs(join(path_out_drop, subdir), exist_ok=True)

    list_ds = tf.data.Dataset.list_files(str(DIR_TEST+'/*/*'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    fp = open(path_out_text, 'a')
    for image, label, file_path in labeled_ds:
        # print("Image shape:", image.numpy().shape)
        # print("Label:", label.numpy().argmax())
        file_path_str = file_path.numpy().decode("utf-8")
        image_name = file_path_str.split('/')[-1]
        print(image_name, end=" --> ")

        prediction = model.predict(tf.expand_dims(image, 0))
        predicted_label = prediction.argmax()
        label_gt = int(label.numpy().argmax())
        if int(predicted_label) == label_gt:
            print('Keep')
            copyfile(file_path_str, join(file_path_str.replace(DIR_TEST, path_out_keep)))
        else:
            print('Drop')
            copyfile(file_path_str, join(file_path_str.replace(DIR_TEST, path_out_drop)))
        ### log to text file
        score = prediction.max()
        fp.write(f'{image_name}\t{label_gt}\t{predicted_label}\t{score}\n')
    fp.close()

    path_make_sh = join(path_out_root, 'make_combine.sh')
    path_out_combine = join(path_out_root, 'combine')

    with open(path_make_sh, 'w') as fp:
        fp.write(f'mkdir combine\n')
        fp.write(f'cp -r {DIR_REAL_DATA}/* {path_out_combine}/\n')
        fp.write(f'cp -r {path_out_keep}/* {path_out_combine}/')


if __name__ == '__main__':
    classifier.summary()
    # model.load_weights(PATH_CHECKPOINT)
    classifier.load_weights(PATH_WEIGHTS)

    test_model(classifier)
