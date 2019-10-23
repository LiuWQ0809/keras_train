from keras.datasets import mnist
import scipy.io as sio
import urllib.request
import shutil
import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import tools.mutil_processing_image_generator_balance as MB
import multiprocessing


def get_data_generator(json_path, classes, batch_size, img_height, img_width, nb_gpu, sample_save_path_training=None):
    pool = multiprocessing.Pool(processes=8)
    datagen = MB.ImageDataGenerator(rescale=1. / 255,
                                        # shear_range=0.2,
                                        # zoom_range=0.1,
                                        # rotation_range = 50,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        pool=pool)
    
    generator = datagen.flow_from_json(
                                    json_path,
                                    target_size=(img_height, img_width),
                                    batch_size=batch_size,
                                    classes=classes,
                                    shuffle=True,
                                    save_to_dir=sample_save_path_training,
                                    class_mode='binary',
                                    nb_gpu=nb_gpu,
                                    is_training=True)
    return generator 


# def get_dataset(dataset='mnist'):
    
#     if dataset=='mnist':
#         (train_x, train_y), (test_x, test_y) = get_mnist()
#     elif dataset=='svhn':
#         (train_x, train_y), (test_x, test_y) = get_svhn()
    
#     train_y = np_utils.to_categorical(train_y, NUM_CLASSES)
#     test_y = np_utils.to_categorical(test_y, NUM_CLASSES)
    
#     return (train_x, train_y), (test_x, test_y)

# if __name__=='__main__':

#     (train_x, train_y), (test_x, test_y) = get_dataset('svhn')
#     print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)

#     (train_x, train_y), (test_x, test_y) = get_dataset('mnist')
#     print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
