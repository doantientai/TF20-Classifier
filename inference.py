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

DIR_TEST = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_20k_generated'
PATH_WEIGHTS = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_1k_baseline/models/weights.09-0.9396.hdf5'


# THRESHOLD_SCORE = 0.21
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

    for image, label, file_name in labeled_ds.take(1):
        print("Image shape:", image.numpy().shape)
        print("Label:", label.numpy().argmax())
        print("File name:", file_name.numpy().decode("utf-8"))

        prediction = model.predict(tf.expand_dims(image, 0))
        print("Predicted label:", prediction.argmax())
        print("Probability:", prediction.max())
    exit()

    # fp = open(path_out_text, 'a')

    # for batch_idx, file_path in enumerate(list_file_paths[3:]):
    #     image = cv2.imread(file_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # print(image.shape)
    #     image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    #     # image = image.transpose(2, 0, 1)
    #     # image = (image / 127.5) - 1.0
    #     # image = torch.from_numpy(image)
    #     # image = image.unsqueeze(0)
    #     # image = image.type(torch.cuda.FloatTensor)
    #     # print(image.min())
    #     # print(image.max())
    #     # print(image)
    #     output = model(image)
    #     _, pred = torch.max(output, 1)
    #
    #     output_np = output.cpu().detach().numpy()
    #     # print(output_np)
    #
    #     output_np = (output_np - output_np.min())
    #     output_np /= output_np.sum()
    #
    #     image_name = file_path.split('/')[-1]
    #     image_source_label = file_path.split('/')[-2]
    #
    #     score = output_np.max()
    #     print(f'{image_name} {score}', end=' ')
    #     pred_int = int(pred.cpu().detach()[0].numpy())
    #
    #     if int(image_source_label) == pred_int:
    #         print(f'--> keep')
    #         copyfile(file_path, join(file_path.replace(DIR_TEST, path_out_keep)))
    #     else:
    #         print(f'--> drop')
    #         copyfile(file_path, join(file_path.replace(DIR_TEST, path_out_drop)))
    #     fp.write(f'{image_name}\t{image_source_label}\t{pred_int}\t{score}\n')
    # fp.close()


if __name__ == '__main__':
    classifier.summary()
    # model.load_weights(PATH_CHECKPOINT)
    classifier.load_weights(PATH_WEIGHTS)

    test_model(classifier)
