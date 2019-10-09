# Test models and evaluation
# Author: Jin Huang
# Date: 08/21/2018
# Date: 09/13/2018 (Adding 29 normal in lymph set)
# Date: 11/02/2018 Test Camelyon16 use models
# Date: 11/08/2018 Re-coding and add more features


import sys
# sys.path.append('/home/jin/my_job/keras-multiprocess-image-data-generator')
import time
import argparse
import os
import numpy as np
import multiprocessing
import keras
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras import applications
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import metrics
from keras.optimizers import *
from keras.utils.data_utils import get_file
# import tools.image_old as T_OLD
import tools.mutil_processing_image_generator_balance as T_OLD
import utilities
from utilities import LossHistory
from utilities import TensorBoardImage
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import h5py
from PIL import Image as pil_image
from sklearn.metrics import auc

os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

try:
    from importlib import reload
    reload(T_OLD)
except:
    reload(T_OLD)

###########################################################
            #Define the parameters#
###########################################################
phase = "test"
# model_type = "inception"
model_type = "resnet"
run_test = True
# run_test = False
# save_wrong_patch = True
save_wrong_patch = False
run_evaluation = True

start = 0.50
end = 0.50
step = 0.05
nb_steps = int(round((end-start)/step)+1)

batch_size = 128
classes = 2
n_process = 1

date = "0711"
base_path = "/home/liu/update/old_Camylon/models/20170711/"

# ResNet
if model_type == "resnet":
    model_name = "resnet50"
    img_list_name = "0404_res50_dz_level_0_balance_gen_img_list_epoch10_70"
    result_npy_name = "0404_res50_dz_level_0_balance_gen_epoch10_70"

elif model_type == "inception":
    model_name = "inception_v4"
    img_list_name = "0404_res50_dz_level_0_normal_gen_img_list"
    result_npy_name = "0404_res50_dz_level_0_normal_gen"

sample_save_path = "/mnt/disk_share/liu_data/lymph_train/sample/0403_bp_3/"
result_save_dir = "/mnt/disk_share/liu_data/lymph_train/result/0403_bp_3/"

###########################################################
            #Set necessary paths#
###########################################################
if model_type == "resnet":
    # model_and_weight_path = "/1TB/jin_data/models/1025_cam_16_17_wild_res18_model_and_weights_epoch_14.hdf5"
    model_and_weight_path = "/mnt/disk_share/liu_data/lymph_train/models/20190520_pre_10_99/0520_dz_res50_level0_normal_gen_0.5_model_and_weights_epoch_10.hdf5"
    nb_samples = 197680
    nb_negtive = 98646
    nb_positive = 99034
    # nb_samples = 2000
    # nb_negtive = 1000
    # nb_positive = 1000
elif model_type == "inception":
    # 0228 DZ model
    nb_samples = 442448
    nb_negtive = 275591
    nb_positive =166857

    model_and_weight_path = "/home/liu/update/old_Camylon/models/20170711/IF_resnet50_0711_model_and_weights_epoch_12.hdf5"
    
    # First fine-tune without freezing and 36 WSIs (level0).
    # model_and_weight_path = "/mnt/data/jin_data/model/0129_fine_tune_inv4_no_freeze_model_and_weights_epoch_5.hdf5"
    # nb_samples = 82084
    # nb_negtive = 33054
    # nb_positive = 49030

    # # Second fine-tune without freezing and 50 WSIs (level0).
    # model_and_weight_path = "/mnt/data/jin_data/model/0206_fine_tune_inv4_level0_no_freeze_model_and_weights_epoch_9.hdf5"
    # # Second fine-tune freeze before block C, 50 WSIs (level0).
    # model_and_weight_path = "/mnt/data/jin_data/model/0207_fine_tune_inv4_level0_release_c_model_and_weights_epoch_9.hdf5"
    # # The only level 1 fine-tune.
    # model_and_weight_path = "/mnt/data/jin_data/model/0207_fine_tune_inv4_level1_no_freeze_model_and_weights_epoch_9.hdf5"

else:
    print("You have to choose a model base between resnet and inception.")
    sys.exit(0)

if model_type == "resnet":
    img_width, img_height = 224, 224

elif model_type == "inception":
    img_width, img_height = 299, 299

else:
    print("You have to choose a model base between resnet and inception.")
    sys.exit(0)

###########################################################
            #Check paths and make dir#
###########################################################
result_save_path = result_save_dir + "/" + result_npy_name
img_list_path = result_save_dir + "/" + img_list_name

if not sample_save_path:
    print("Path does not exist. Creating model saving directory ...")
    os.mkdir(sample_save_path)

if not result_save_dir:
    print("Path does not exist. Creating prediction saving directory ...")
    os.mkdir(result_save_dir)

if phase == "debug":
    # Only have 299 size for debug now
    test_set_dir = "/30TB/jin_data/cam16_processed/debug_1029_valid_299"
    # Reset the nb samples for debugging
    nb_samples = None
    nb_negtive = None
    nb_positive = None

elif phase == "test":
    if model_type == "resnet":
        test_set_dir = "/mnt/disk_share/liu_data/lymph_train/json/valset_fix_size/"
        # test_set_dir = "/mnt/disk_share/data/breast_cancer/lymph_node/intraoperative_frozen/debug/224_0_1/" # Just for debug
    elif model_type == "inception":
        test_set_dir = "/mnt/data/jin_data/lymph_private/0211_dz_processed/valid_299/level0"
    else:
        print("You have to choose a model base between resnet and inception.")
        sys.exit(0)

else:
    print("You have choose between debug and test!")
    sys.exit(0)

# if self.save_to_dir:
#             for i in range(current_batch_size):
#                 img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=current_index + i,
#                                                                   hash=np.random.randint(1e4),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))

#######################################################################################
                            #Load model and use it for test#
#######################################################################################
if run_test:
    print("Run test process...")

    # Tensorflow GPU management
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

    # Load the model
    model = load_model(model_and_weight_path)
    classes = 2
    # Define data generator
    test_datagen = T_OLD.ImageDataGenerator(rescale=1. / 255,
                                            pool= multiprocessing.Pool(processes=n_process))
    # test_generator = test_datagen.flow_from_directory(
    #                                     test_set_dir,
    #                                     target_size=(img_height, img_width),
    #                                     batch_size=batch_size,
    #                                     class_mode='binary',
    #                                     shuffle=False,
    #                                     save_to_dir=sample_save_path,
    #                                     seed=None,
    #                                     phase="test",
    #                                     save_list_dir=img_list_path,
    #                                     nb_gpu=1)
    test_generator = test_datagen.flow_from_json(
                                    test_set_dir,
                                    target_size=(img_height, img_width),
                                    batch_size=batch_size,
                                    classes=classes,
                                    shuffle=False,
                                    save_to_dir=sample_save_path,
                                    class_mode='binary',
                                    nb_gpu=1,
                                    is_training=False)

    probabilities = model.predict_generator(test_generator,
                                            steps=nb_samples//batch_size,
                                            verbose=1)

    print(max(probabilities))
    # sys.exit(0)
    print("Prediction finished!")
    print("Saving prediction into npy file...")
    np.save(result_save_path, probabilities)
    print("Results saved!")

else:
    print("[INFO] Only processing the predictions.")

####################################################################
        #Calculate the evaluation according to thresholds#
####################################################################
if run_evaluation:
    print("[INFO] Loading predictions ...")
    # result_np = np.load(result_save_path + ".npy")
    # result_np = np.load("/mnt/data/jin_data/results/0212_inv4_25wsi_level_0_no_freeze_valid_8w.npy")
    # img_list = np.load("/mnt/data/jin_data/results/0212_inv4_25wsi_no_freeze_valid_8w_img_list.npy")
    result_np = np.load(result_save_path + '.npy') #"/mnt/disk_share/liu_data/lymph_train/result/0228_res50_dz_level_0_normal_gen_epoch4.npy")
    img_list = np.load(img_list_path + '.npy') #"/mnt/disk_share/liu_data/lymph_train/result/0228_res50_dz_level_0_normal_gen_img_list_epoch4.npy")
    # result_np = np.load("/mnt/data/jin_data/results/0212_inv4_50wsi_level_0_from_c_valid_8w.npy")

    # Calculate the nb of prediction
    quotient = nb_samples // batch_size
    nb_excess_patch = nb_samples - quotient*batch_size
    nb_sample_final = nb_samples - nb_excess_patch

    # Confirm number of samples
    if nb_sample_final != result_np.shape[0]:
        print("nb_sample for prediction: %f." % nb_sample_final)
        print("nb_sample obtained from npy: % f" % result_np.shape[0])
        print("Please check the number of samples!")
        sys.exit(0)
    else:
        print("Number of samples matches, starting evaluation ...")

    # Get the image path list according to nb_samples
    print("[INFO] Loading image path list ...")
    # img_list = np.load(img_list_path + ".npy")
    img_list = np.array(img_list[0: nb_sample_final])

    # AUC x axis: FP
    # AUC y axis: TF
    all_fp = []
    all_tp = []
    all_tn = []
    all_fn = []
    all_thres = []
    all_recall = []
    all_recall.append(1)
    all_precision = []
    all_acc = []
    all_fpr = []
    all_fpr.append(1)


    for c in np.linspace(start, end, nb_steps):
        print("*************************************************************")
        c = round(c, 2)
        cut_off_thres = c
        print("Cut off threshold: %f" % cut_off_thres)

        # Count the normal part
        count_negtive = 0
        true_negative = 0
        false_positive = 0

        for i in range(nb_negtive):
            pred = result_np[i]
            if pred < cut_off_thres:
                true_negative += 1
            else:
                false_positive += 1

                if save_wrong_patch == True:
                    # save the img for the wrong prediction
                    img_path = img_list[i]
                    img = pil_image.open(test_set_dir + "/" + img_path)

                    # false_positive_save_path = base_path + "/" + date + "_" + model_name
                    # if not os.path.isdir(false_positive_save_path):
                    #     print("Path does not exist. Creating directory ...")
                    #     os.mkdir(false_positive_save_path)

                    # false_positive_save_path = false_positive_save_path + "/" + str(cut_off_thres*100)
                    # false_positive_save_path = "/mnt/data/jin_data/samples/wrong_preds/36_no_freeze_inv4/" + str(cut_off_thres * 100)
                    false_positive_save_path = result_save_dir + str(cut_off_thres * 100)
                    if not os.path.isdir(false_positive_save_path):
                        print("Path does not exist. Creating directory ...")
                        os.mkdir(false_positive_save_path)

                    false_positive_save_path = false_positive_save_path + "/" + "fp"
                    if not os.path.isdir(false_positive_save_path):
                        print("Path does not exist. Creating directory ...")
                        os.mkdir(false_positive_save_path)

                    img.save(false_positive_save_path + "/" + img_list[i][2:])

        print("[INFO] True negative: %d" % true_negative)
        print("[INFO] False positive: %d" % false_positive)

        # Count the tumor part
        count_positive = 0
        true_positive = 0
        false_negative = 0

        for i in range(nb_negtive, result_np.shape[0]):
            pred = result_np[i]
            if pred > cut_off_thres:
                true_positive += 1
            else:
                false_negative += 1

                if save_wrong_patch == True:
                    # save the img for the wrong prediction
                    img_path = test_set_dir + "/" + img_list[i]
                    img = pil_image.open(img_path)

                    # false_negative_save_path = base_path + "/" + date + "_" + model_name
                    # if not os.path.isdir(false_negative_save_path):
                    #     print("Path does not exist. Creating directory ...")
                    #     os.mkdir(false_negative_save_path)

                    # false_negative_save_path = false_negative_save_path + "/" + str(cut_off_thres * 100)
                    # false_negative_save_path = "/mnt/data/jin_data/samples/wrong_preds/36_no_freeze_inv4/" + str(cut_off_thres * 100)
                    false_negative_save_path = result_save_dir + str(cut_off_thres * 100)
                    if not os.path.isdir(false_negative_save_path):
                        print("Path does not exist. Creating directory ...")
                        os.mkdir(false_negative_save_path)

                    false_negative_save_path = false_negative_save_path + "/" + "fn"
                    if not os.path.isdir(false_negative_save_path):
                        print("Path does not exist. Creating directory ...")
                        os.mkdir(false_negative_save_path)

                    img.save(false_negative_save_path + "/" + img_list[i][2:])

        print("[INFO] True positive: %d" % true_positive)
        print("[INFO] False negative: %d" % false_negative)

        all_fn.append(false_negative)
        all_fp.append(false_positive)
        all_tn.append(true_negative)
        all_tp.append(true_positive)
        all_thres.append(cut_off_thres)

        # Evaluation metrics
        # Recall = TP/(TP+FN)
        recall = (float(true_positive)/(float(true_positive)+float(false_negative)))
        print("Recall (AKA: sensitivity or True Positive Rate): %f" % recall)

        # Precision = TP/(TP+FP)
        if true_positive == 0:
            precision = 0
        else:
            precision = (float(true_positive)/(float(true_positive)+float(false_positive)))
        print("Precision (AKA: specificity): %f" % precision)

        # Accuracy = (TP+TN)/(TP+TN+FP+FN)
        acc = ((float(true_positive)+float(true_negative))/float(result_np.shape[0]))
        print("Accuracy: %f" % acc)

        # F1 score = 2*precision*recall/(precision+recall)
        f_score = 2*precision*recall/(precision+recall)
        print("F1-score is %f" % f_score)

        false_positive_rate = float(false_positive)/(float(false_positive)+float(true_negative))

        all_recall.append(recall)
        all_precision.append(precision)
        all_fpr.append(false_positive_rate)

    # Calculate AUC
    all_recall.append(0)
    all_fpr.append(0)
    all_recall = np.array(all_recall)
    all_fpr = np.array(all_fpr)

    print("************************")
    auc = auc(all_fpr, all_recall)
    print("[!] AUC is: %f" % auc)

else:
    print("No evaluation processed. Set 'run_evaluation' to True if you need the evaluation results.")