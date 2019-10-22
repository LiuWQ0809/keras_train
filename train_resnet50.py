# Training pipeline for resnet50
# Author: Jin Huang
# Date: 09/20/2018 ver_1.0

import numpy as np
import keras
import sys
import os
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import model_from_json, load_model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import tools.image_old as T_OLD
import multiprocessing
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.utils.multi_gpu_utils import multi_gpu_model
import argparse
from keras import metrics
from keras.optimizers import *
from keras.callbacks import TensorBoard
from utilities import LossHistory
from utilities import TensorBoardImage
import utilities
from utilities import MyCbk
# from keras.applications.resnet50 import ResNet50
import resnet_factory # import ResNet50
# import tools.threshold_choose_preprocesing as JH
import tools.mutil_processing_image_generator_balance as MB
# import feature_map as FM

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

import time
import pdb

np.random.seed(2019)
tf.set_random_seed(2019)

# try:
#     from importlib import reload
#     reload(T)
# except:
#     reload(T)

try:
    pool.terminate()
except:
    pass

os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning 

model_path = "/home/liu/work/models/postoperative_resnet50_0923_model_and_weights_epoch_21.hdf5"
# model_path = "/mnt/disk_share/liu_data/lymph_train/models/IF_resnet50_20190930_4class_4s/IF_resnet50_20190930_4class_4s_model_and_weights_epoch_12.hdf5"
# model_path = "/mnt/disk_share/liu_data/lymph_train/models/20190920_4class_4s/IF_resnet50_0920_weights_only_epoch_8.h5"
# model_path = "/home/liu/update/old_Camylon/models/20190713_0805_3class_all/IF_resnet50_0805_model_and_weights_epoch_12.hdf5"
# model_path = "/mnt/disk_share/liu_data/lymph_train/models/20190713_0805_3class_all/IF_resnet50_0805_weights_only_epoch_12.h5"
# model_path = "/home/liu/update/old_Camylon/models/20190713_0801_froze_conv/IF_resnet50_0713_0801_0107_model_and_weights_epoch_7.hdf5"
# model_path = "/home/liu/update/old_Camylon/models/20170714/IF_resnet50_0714_model_and_weights_epoch_4.hdf5"
# model_path = "/home/liu/update/old_Camylon/models/20190713_0801_froze_conv/IF_resnet50_0713_0801_0107_model_and_weights_epoch_1.hdf5"
###########################################################
            #Define the parameters#
###########################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
img_width, img_height = 224, 224
batch_size = 8
epochs = 30
classes = 2
activation = 'binary'
nb_gpu = 2
n_process = 16
pool = multiprocessing.Pool(processes=n_process)
dropout_keep_prob = 0.8
weights = "imagenet"
network = "50"
validation_format = "fix"

pooling = 'max'
class_weight = False
FL = False
phase = "train"
log_dir = "./debug/logs/"
model_dir = "./debug/models/"
model_name = "debug_keras_1017"
float_16 = False

if not model_dir:
    print("Path does not exist. Creating TensorBoard saving directory ...")
    os.mkdir(model_dir)

if not log_dir:
    print("Path does not exist. Creating TensorBoard saving directory ...")
    os.mkdir(log_dir)

if float_16:
    from keras import backend as K
    print(K.floatx())
    K.set_floatx('float16')
    print(K.floatx())

#######################################################
# Provide the directory for the json files#
#######################################################
"""
For training set:
    Always use random crop and flow from the jsons.

For validation set:
    If you choose to use random crop for validation as it is in training,
    you should set the the validation_format to "random" and
    provide the path for validation json files.

    If you choose to use a fixed cropped set (image size 299*299 for inception v3 and v4),
    you should set the the validation_format to "fix" and
    provide the directory for validation images.

"""

# Path for the saving and debugging images
sample_save_path_training = "/1TB/jin_data/samples/0116"
sample_save_path_valid = "/1TB/jin_data/samples/0116"

# Original nb_sample and paths for training
# training_json_path_for_training = "/mnt/disk_share/liu_data/lymph_train/json/debug/"
training_json_path_for_training = "/14TB/liu_data/debug/"# "/mnt/disk_share/liu_data/lymph_train/json/json_hard_example_0925/"# "/mnt/disk_share/liu_data/lymph_train/json/json_hard_example_0827/"# "/mnt/disk_share/liu_data/lymph_train/json/json_only_clean_0715/"# "/mnt/disk_share/liu_data/lymph_train/json/json_hard_example_0731/"# "/mnt/disk_share/liu_data/lymph_train/json/json_select_100_clean_0604/"
# validation_set_dir_for_training = "/mnt/disk_share/liu_data/lymph_train/json/valset_fix_size_debug/"
validation_set_dir_for_training = "/mnt/disk_share/liu_data/lymph_train/json/valset_fix_size/"
# validation_set_dir_for_training = "/mnt/disk_share/data/breast_cancer/lymph_node/intraoperative_frozen/debug/224_debug/"
nb_training_samples_train_init = 1000
# nb_random_validation_samples_train_init = 13388
nb_fix_validation_samples_train_init = 1800

# Original nb_sample and paths for debugging
# training_json_path_for_debugging = "/30TB/jin_data/cam16_processed/0114_debug/json"
# validation_json_path_for_debugging = None
# validation_set_dir_for_debugging = "/30TB/jin_data/cam16_processed/valid_224/0117_filtered"

# nb_training_samples_debug_init = 14112
# nb_random_validation_samples_debug_init = None
# nb_fix_validation_samples_debug_init = 351820
# nb_fix_validation_samples_debug_init = 24497


####################################################################################
        # Adapt nb of samples for training/debug for single or multi-GPU #
####################################################################################
def calculate_nb_sample(batch_size, initial_nb_sample, nb_gpu):
    multiple = nb_gpu * batch_size
    quotient = initial_nb_sample // multiple
    nb_excess_patch = initial_nb_sample - quotient * multiple
    final_nb_sample = initial_nb_sample - nb_excess_patch

    return final_nb_sample

training_json_path = training_json_path_for_training
nb_training_sample = calculate_nb_sample(batch_size=batch_size,
                                            initial_nb_sample=nb_training_samples_train_init,
                                            nb_gpu=nb_gpu)
validation_set_dir = validation_set_dir_for_training
nb_validation_sample = calculate_nb_sample(batch_size=batch_size,
                                            initial_nb_sample=nb_fix_validation_samples_train_init,
                                            nb_gpu=nb_gpu)


###########################################################
                #Multi-GPU settings#
###########################################################
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=nb_gpu,
                help="# of GPUs to use for training")
args = vars(ap.parse_args())
# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]


# ###########################################################
#                         #Select model#
# ##########################################################
# """
# Model options: "50", "se_50", "101", "152"

# """
if network == "50":
    base_model = resnet_factory.ResNet50(include_top=False, weights=None) #, x1, x2, x3, x4, x5
# elif network == "se_50":
#     base_model = resnet_factory.se_ResNet50(include_top=False, weights='imagenet')
# elif network == "101":
#     base_model = resnet_factory.ResNet101(include_top=False, weights=None)
# elif network == "152":
#     base_model = resnet_factory.ResNet152(include_top=False, weights=None)
# else:
#     print("Please select a valid network for training!")
#     sys.exit(0)

# base_model

# model = load_model(model_path)
# pdb.set_trace()
# base_model_out = base_model.output
# base_model_out = keras.layers.GlobalAveragePooling2D()(base_model_out)


base_model_out = base_model.output
# base_model_out
if pooling == 'max':
    base_model_out = keras.layers.GlobalMaxPooling2D()(base_model_out)
elif pooling == 'ave':
    base_model_out = keras.layers.GlobalAveragePooling2D()(base_model_out)
predictions = Dense(1, activation='sigmoid')(base_model_out)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(model_path, by_name=True)
parallel_model = model
###########################################################
                        #Build model#
###########################################################
# # Choose logistic layer according to the loss function.
# if activation == 'binary': # Default binary
#     predictions = Dense(1, activation='sigmoid')(base_model_out)
#     if G <= 1:
#         print("[INFO] training with 1 GPU...")
#         model = Model(inputs=base_model.input, outputs=predictions)
#         # model.summary()
#         parallel_model = model
#         parallel_model.compile(optimizer='adam',
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])
#     else:
#         print(("[INFO] training with {} GPUs...".format(G)))
#         with tf.device('/cpu:0'):
#             model = Model(inputs=base_model.input, outputs=predictions)

#         parallel_model = multi_gpu_model(model, gpus=G)
#         parallel_model.compile(optimizer='adam',
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])

# elif activation == 'softmax': # Default softmax
#     print("Using normal softmax cross entropy as loss ...")
#     predictions = Dense(2, activation='softmax')(base_model_out)
#     if G <= 1:
#         print("[INFO] training with 1 GPU...")
#         model = Model(inputs=base_model.input, outputs=predictions)
#         # model.summary()
#         # print model.summary()
#         parallel_model = model
#         parallel_model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=[metrics.categorical_accuracy])
#     else:
#         print(("[INFO] training with {} GPUs...".format(G)))
#         with tf.device('/cpu:0'):
#             model = Model(inputs=base_model.input, outputs=predictions)

#         parallel_model = multi_gpu_model(model, gpus=G)
#         parallel_model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=[metrics.categorical_accuracy])

# else:
#     print("You have to choose the activation between sigmoid and softmax!")
#     sys.exit(0)

# print("********************training**************************")

# model = load_model(model_path)
# pdb.set_trace()
# for layer in model.layers[:-34]: # -3: just update fc layer     -34: update stage5 and fc layer  -96: update stage4  stage5 and fc layer
#     # print(layer.trainable)
#     layer.trainable = False
# import pdb
# pdb.set_trace()
# parallel_model = parallel_model.load_weights(model_path)



parallel_model = multi_gpu_model(model, gpus=G)
if FL:
    parallel_model.compile(optimizer='adam',
                    loss=utilities.binary_focal_loss(gamma=0., alpha=.5),
                    metrics=['accuracy', utilities.fmeasure, utilities.recall, utilities.precision, utilities.auc])
else:
    parallel_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', utilities.fmeasure, utilities.recall, utilities.precision, utilities.auc])
###########################################################
                    #Data generator#
###########################################################
# Data Generator for training (always use json files)
train_datagen = MB.ImageDataGenerator(rescale=1. / 255,
                                        # shear_range=0.2,
                                        # zoom_range=0.1,
                                        # rotation_range = 50,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        pool = pool)

training_generator = train_datagen.flow_from_json(
                                    training_json_path,
                                    target_size=(img_height, img_width),
                                    batch_size=batch_size,
                                    classes=classes,
                                    shuffle=True,
                                    save_to_dir=sample_save_path_training,
                                    class_mode='binary',
                                    nb_gpu=nb_gpu,
                                    is_training=True)

# print("INFO  train data ###########################################################################")


# valid_datagen = MB.ImageDataGenerator(rescale=1. / 255,
#                                             pool=pool)

# validation_generator = valid_datagen.flow_from_json(
#                                     validation_set_dir,
#                                     target_size=(img_height, img_width),
#                                     batch_size=batch_size,
#                                     classes=classes,
#                                     shuffle=True,
#                                     save_to_dir=sample_save_path_valid,
#                                     class_mode='binary',
#                                     nb_gpu=nb_gpu,
#                                     is_training=False)


###########################################################
                    #Callback functions#
###########################################################
loss_history = LossHistory()
lrate = LearningRateScheduler(utilities.step_decay)
cbk = MyCbk(model, model_dir, model_name)

tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32,
                            write_graph=True, write_grads=False, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# fp_callback=FM.MyTensorBoard(log_dir=log_dir,input_images=training_generator.__next__(), batch_size=batch_size,
                             # update_features_freq=64,write_features=True,write_graph=True,update_freq='batch')


callbacks_list = [loss_history, lrate, cbk, tb_callback]

###########################################################
                        #Training#
###########################################################
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# if class_weight:
#     # Set your class weight here!
#     class_weight = {0:1., 1:65.40}

#     print("=>Start training with class weights ...")
#     print("Choice of loss:" + loss_fun)
#     parallel_model.fit_generator(generator=training_generator,
#                         steps_per_epoch=nb_training_sample//(batch_size*nb_gpu),
#                         validation_data=validation_generator,
#                         epochs=epochs,
#                         use_multiprocessing=False,
#                         workers=1,
#                         validation_steps=nb_validation_sample//(batch_size*nb_gpu),
#                         verbose=1,
#                         class_weight=class_weight,
#                         callbacks=callbacks_list)
# else:
#     print("=>Start training without class weight...")
#     print("Choice of activation:" + activation)
#     parallel_model.fit_generator(generator=training_generator,
#                     steps_per_epoch=nb_training_sample//(batch_size),#*nb_gpu),
#                     # validation_data=validation_generator,
#                     epochs=epochs,
#                     use_multiprocessing=False,
#                     workers=1,
#                     # validation_steps=nb_validation_sample//(batch_size),#*nb_gpu),
#                     verbose=1,
#                     callbacks=callbacks_list)

# print("=> Whole training process finished.")


# def write_log(callback, names, logs, batch_no):
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()
    
# net_in = Input(shape=(3,))
# net_out = Dense(1)(net_in)
# model = Model(net_in, net_out)
# model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
 
# log_path = './graph'
# callback = TensorBoard(log_path)
# callback.set_model(model)
# train_names = ['train_loss', 'train_mae']
# val_names = ['val_loss', 'val_mae']
start_time = time.time()
X_train, Y_train = next(training_generator)
for batch_no in range(100):
    # pdb.set_trace()
    logs = parallel_model.train_on_batch(X_train, Y_train)
    # pred_label = parallel_model.predict(X_train)
    # pred_label[pred_label > 0.5] = 1
    # pre_labels = np.int8(pred_label)
    # logs = parallel_model.fit(x_train, y_train, batch_size=32, epochs=10)
    # write_log(callback, train_names, logs, batch_no)
    
    # if batch_no % 10 == 0:
    #     X_val, Y_val = np.random.rand(32, 3), np.random.rand(32, 1)
    #     logs = model.train_on_batch(X_val, Y_val)
End_time = time.time() - start_time
print("INFO train times = {}".format(End_time))

