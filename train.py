from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
from shutil import copyfile

# DIR_TRAIN = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_20k_generated_split/combine'
# DIR_VALID = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
# DIR_PROJECT = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_1k_20k_selected'

# DIR_TRAIN = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_10k_generated_split/combine'
# DIR_VALID = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
# DIR_PROJECT = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_1k_10k_selected'

DIR_TRAIN = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA_1k'
DIR_VALID = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
DIR_PROJECT = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_1k_baseline_augmentation'

# DIR_TRAIN = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA_128'
# DIR_VALID = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
# DIR_PROJECT = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_128_baseline'

# DIR_TRAIN = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA'
# DIR_VALID = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
# DIR_PROJECT = '/media/tai/6TB/Projects/TF20/Classifier/Projects/train_001_A_10_baseline_categorical_loss'
EARLY_STOP = 3  # Number of waiting epochs or None

# EARLY_STOP = None  # Number of waiting epochs or None

BATCH_SIZE = 512
VAL_BATCH_SIZE = 1000
EPOCHS = 10
IMG_HEIGHT = IMG_WIDTH = 32

SUMMARY = False

num_train_samples_per_epoch = 1000000
num_valid_samples_per_epoch = 10000

if __name__ == '__main__':
    os.makedirs(DIR_PROJECT)
    copyfile(os.path.join(os.getcwd(), 'train.py'),
             os.path.join(DIR_PROJECT, 'train.py'))

    list_classes = [str(i) for i in range(10)]
    num_classes = len(list_classes)
    print(f'Categories:')
    print(list_classes)

    train_image_generator = ImageDataGenerator(
        rescale=1./255
    )  # Generator for our training data

    valid_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=DIR_TRAIN,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical',
                                                               classes=list_classes)

    valid_data_gen = valid_image_generator.flow_from_directory(batch_size=VAL_BATCH_SIZE,
                                                               directory=DIR_VALID,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical',
                                                               classes=list_classes)

    sample_training_images, _ = next(train_data_gen)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if SUMMARY:
        model.summary()

    dir_tensorboard = os.path.join(DIR_PROJECT, 'logs')
    dir_save_models = os.path.join(DIR_PROJECT, 'models')
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=dir_tensorboard),
        # save model
        tf.keras.callbacks.ModelCheckpoint(filepath=dir_save_models)
    ]
    # Interrupt training if `val_loss` stops improving for over EARLY_STOP epochs
    if EARLY_STOP is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=EARLY_STOP, monitor='val_loss'))

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=num_train_samples_per_epoch // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_data_gen,
        validation_steps=num_valid_samples_per_epoch // BATCH_SIZE,
        callbacks=callbacks
    )
