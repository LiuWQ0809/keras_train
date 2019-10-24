from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import np_utils
import keras.backend as K
import keras
from keras_applications.imagenet_utils import _obtain_input_shape

import tensorflow as tf
from datasets import get_data_generator

import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import argparse

import tools.mutil_processing_image_generator_balance as MB
import resnet_factory
from GRL import GradientReversal

import pdb


class ADDA():
    def __init__(self, args):
        # Input shape
        self.args = args
        self.img_shape = args.processed_size
        self.src_flag = False
        self.disc_flag = False
        self.grl_layer = GradientReversal(1.0)
        
        self.discriminator_decay_rate = 3 #iterations
        self.discriminator_decay_factor = 0.5
        self.src_optimizer = Adam(args.lr, 0.5)
        self.tgt_optimizer = Adam(args.lr, 0.5)
        
    def define_source_encoder(self, weights=None):
    
        #self.source_encoder = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=self.img_shape, pooling=None, classes=10)
        # inp = Input(shape=self.img_shape)

        base_model = resnet_factory.ResNet50(include_top=False, weights=None, input_shape=self.img_shape)
        base_model_out = base_model.output
        # x = MaxPooling2D(pool_size=(2, 2))(base_model_out)
        # pdb.set_trace()
        self.source_encoder = Model(inputs=base_model.input, outputs=base_model_out)
        
        self.src_flag = True
        
        if weights is not None:
            self.source_encoder.load_weights(weights, by_name=True)
    
    def define_target_encoder(self, weights=None):
        
        if not self.src_flag:
            self.define_source_encoder()
        
        with tf.device('/cpu:0'):
            self.target_encoder = clone_model(self.source_encoder)
        
        if weights is not None:
            self.target_encoder.load_weights(weights, by_name=True)
        
    def get_source_classifier(self, model, weights=None):
        
        # x = Flatten()(model.output)
        # x = Dense(128, activation='relu')(x)
        base_model_out = keras.layers.GlobalAveragePooling2D()(model.output)
        x = Dense(1, activation='sigmoid')(base_model_out)
        
        source_classifier_model = Model(inputs=(model.input), outputs=(x))
        # pdb.set_trace()
        if weights is not None:
            source_classifier_model.load_weights(weights)
        
        return source_classifier_model
    
    def define_discriminator(self, model):
        input_shape = model.output_shape[1:]
        inp = Input(input_shape)
        pdb.set_trace()
        x = Flatten()(inp)
        # x = tf.negative(x)
        
        feature_output_grl = self.grl_layer(x)
        x = Dense(128, activation=LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(0.01), name='discriminator1')(feature_output_grl)
        
        predictions = Dense(2, activation='sigmoid', name='discriminator2')(x)
        
        self.disc_flag = True
        # pdb.set_trace()
        self.discriminator_model = Model(inputs=(inp), outputs=(predictions), name='discriminator')
    
    def tensorboard_log(self, callback, names, logs, batch_no):
        
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    
    def get_discriminator(self, model, weights=None):
        
        if not self.disc_flag:
            self.define_discriminator(model)
        pdb.set_trace()
        disc = Model(inputs=(model.input), outputs=(self.discriminator_model(model.output)))
        
        if weights is not None:
            disc.load_weights(weights, by_name=True)
        
        return disc
      
    def train_source_model(self, model, epochs=2, batch_size=32, save_interval=1, start_epoch=0):

        train_generator = get_data_generator(self.args.src_trainset,
                                                classes=2,
                                                batch_size=batch_size,
                                                img_height=self.img_shape[0],
                                                img_width=self.img_shape[1],
                                                nb_gpu=self.args.num_gpus)
        val_generator = get_data_generator(self.args.src_valset,
                                                classes=2,
                                                batch_size=batch_size,
                                                img_height=self.img_shape[0],
                                                img_width=self.img_shape[1],
                                                nb_gpu=self.args.num_gpus)

        model.compile(loss='binary_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])
        
        if not os.path.isdir('data'):
            os.mkdir('data')
        
        saver = keras.callbacks.ModelCheckpoint('data/encoder_{epoch:02d}.hdf5', 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=False, 
                                        save_weights_only=True, 
                                        mode='auto', 
                                        period=save_interval)

        scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, verbose=0, mode='min')

        if not os.path.isdir('data/tensorboard'):
            os.mkdir('data/tensorboard')
    
        visualizer = keras.callbacks.TensorBoard(log_dir=os.path.join('data/tensorboard'), 
                                            histogram_freq=0, 
                                            write_graph=True, 
                                            write_images=False)
        
        model.fit_generator(train_generator,
                            steps_per_epoch=10, 
                            epochs=epochs,
                            callbacks=[saver, scheduler, visualizer], 
                            validation_data=val_generator,
                            validation_steps=10,
                            initial_epoch=start_epoch)
        
    def train_target_discriminator(self, source_model=None, src_discriminator=None, tgt_discriminator=None, epochs=2000, batch_size=100, save_interval=1, start_epoch=0, num_batches=100):   
    
        source_generator = get_data_generator(self.args.src_trainset,
                                                classes=2,
                                                batch_size=batch_size,
                                                img_height=self.img_shape[0],
                                                img_width=self.img_shape[1],
                                                nb_gpu=self.args.num_gpus)
        target_generator = get_data_generator(self.args.tgt_trainset,
                                                classes=2,
                                                batch_size=batch_size,
                                                img_height=self.img_shape[0],
                                                img_width=self.img_shape[1],
                                                nb_gpu=self.args.num_gpus)
        self.define_source_encoder(source_model)
                
        for layer in self.source_encoder.layers:
            layer.trainable = False
        
        source_discriminator = self.get_discriminator(self.source_encoder, src_discriminator)
        # pdb.set_trace()
        target_discriminator = self.get_discriminator(self.target_encoder, tgt_discriminator)
        
        if src_discriminator is not None:
            source_discriminator.load_weights(src_discriminator, by_name=True)
        if tgt_discriminator is not None:
            target_discriminator.load_weights(tgt_discriminator, by_name=True)
        
        source_discriminator.compile(loss = "binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])
        target_discriminator.compile(loss = "binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])
        
        callback1 = keras.callbacks.TensorBoard('data/tensorboard')
        callback1.set_model(source_discriminator)
        callback2 = keras.callbacks.TensorBoard('data/tensorboard')
        callback2.set_model(target_discriminator)
        src_names = ['src_discriminator_loss', 'src_discriminator_acc']
        tgt_names = ['tgt_discriminator_loss', 'tgt_discriminator_acc']

        source, source_label = source_generator.next()
        target, target_label = target_generator.next()
        
        for iteration in range(start_epoch, epochs):
            
            avg_loss, avg_acc, index = [0, 0], [0, 0], 0
            for step in range(0, 100):
                # pdb.set_trace()
                loss1, acc1 = source_discriminator.train_on_batch(source, np_utils.to_categorical(np.zeros(source.shape[0]), 2))
                loss2, acc2 = target_discriminator.train_on_batch(target, np_utils.to_categorical(np.ones(target.shape[0]), 2))
                index+=1
                loss, acc = (loss1+loss2)/2, (acc1+acc2)/2
                print (iteration+1,': ', index,'/', num_batches, '; Loss: %.4f'%loss, ' (', '%.4f'%loss1, '%.4f'%loss2, '); Accuracy: ', acc, ' (', '%.4f'%acc1, '%.4f'%acc2, ')')
                avg_loss[0] += loss1
                avg_acc[0] += acc1
                avg_loss[1] += loss2
                avg_acc[1] += acc2
                if index%num_batches == 0:
                    break
            
            if iteration%self.discriminator_decay_rate==0:
                lr = K.get_value(source_discriminator.optimizer.lr)
                K.set_value(source_discriminator.optimizer.lr, lr*self.discriminator_decay_factor)
                lr = K.get_value(target_discriminator.optimizer.lr)
                K.set_value(target_discriminator.optimizer.lr, lr*self.discriminator_decay_factor)
                print ('Learning Rate Decayed to: ', K.get_value(target_discriminator.optimizer.lr))
            
            if iteration%save_interval==0:
                source_discriminator.save_weights('data/discriminator_source_%02d.hdf5'%iteration)
                target_discriminator.save_weights('data/discriminator_target_%02d.hdf5'%iteration)
                
            self.tensorboard_log(callback1, src_names, [avg_loss[0]/3200, avg_acc[0]/3200], iteration)
            self.tensorboard_log(callback2, tgt_names, [avg_loss[1]/3200, avg_acc[1]/3200], iteration)
    
    def eval_source_classifier(self, model, dataset='Source', batch_size=128, domain='Source'):
        
        # train_generator = get_data_generator(json_path, classes, batch_size, img_height, img_width, sample_save_path_training, nb_gpu)
        val_generator = get_data_generator(self.args.src_valset,
                                            classes=2,
                                            batch_size=batch_size,
                                            img_height=self.img_shape[0],
                                            img_width=self.img_shape[1],
                                            nb_gpu=self.args.num_gpus)
                      
        model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])

        scores = model.evaluate_generator(val_generator.next(),10000)
        print('%s %s Classifier Test loss:%.5f'%(dataset.upper(), domain, scores[0]))
        print('%s %s Classifier Test accuracy:%.2f%%'%(dataset.upper(), domain, float(scores[1])*100))            
            
    def eval_target_classifier(self, source_model, target_discriminator, dataset='Target'):
        
        self.define_target_encoder()
        model = self.get_source_classifier(self.target_encoder, source_model)
        model.load_weights(target_discriminator, by_name=True)
        model.summary()
        self.eval_source_classifier(model, dataset=dataset, domain='Target')

def args_define():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_weights', required=False, help="Path to weights file to load source model for training classification/adaptation")
    parser.add_argument('-e', '--start_epoch', type=int,default=1, required=False, help="Epoch to begin training source model from")
    parser.add_argument('-n', '--discriminator_epochs', type=int, default=10000, help="Max number of steps to train discriminator")
    parser.add_argument('-l', '--lr', type=float, default=0.0001, help="Initial Learning Rate")
    parser.add_argument('-f', '--train_discriminator', action='store_true', help="Train discriminator model (if TRUE) vs Train source classifier")
    parser.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    parser.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    parser.add_argument('-t', '--eval_source_classifier', default=None, help="Path to source classifier model to test/evaluate")
    parser.add_argument('-d', '--eval_target_classifier', default=None, help="Path to target discriminator model to test/evaluate")
    parser.add_argument("--tgt_trainset", type=str, default="/mnt/share/liu_data/lymph_data/json_IF/debug/", help="target train base folder path")
    parser.add_argument("--tgt_valset", type=str, default="/mnt/share/liu_data/lymph_data/json_IF/debug/", help="target val base folder path")
    parser.add_argument("--src_trainset", type=str, default="/mnt/share/liu_data/lymph_data/json_IF/debug/", help="source train base folder path")
    parser.add_argument("--src_valset", type=str, default="/mnt/share/liu_data/lymph_data/json_IF/debug/", help="source val base folder path")
    parser.add_argument("--nb_per_class", type=list, default=[1, 1], help="the per class sample number of training")
    parser.add_argument("--num_threads_batch", type=float, default=[10, 40], nargs=2, help="numbers of threads for data queue")
    parser.add_argument("--labels", type=list, default=["normal", 'tumor'], help="the labels of data")
    parser.add_argument("--foreground_rate_per_class", type=float, default=[0.1, 0.7], nargs=2, help="foreground rate per class of training")
    parser.add_argument("--non_zero_rate", type=float, default=0.99, help="mask rate per class of training")
    parser.add_argument('--gpu_select', default='7', action='store', help='Set which gpu devices to use')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_define()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_select  # "7" # Please don't add spaces.
    args.source_discriminator_weights = "/home/liu/work/models/postoperative_resnet50_0923_model_and_weights_epoch_21.hdf5"
    args.target_discriminator_weights = "/home/liu/work/models/postoperative_resnet50_0923_model_and_weights_epoch_21.hdf5"
    # Dataset path
    args.tgt_trainset = "/14TB/liu_data/debug/"
    args.tgt_valset = "/14TB/liu_data/debug/"
    args.src_trainset = "/14TB/liu_data/debug/"
    args.src_valset = "/14TB/liu_data/debug/"
    args.train_discriminator = True

    args.num_gpus = int((len(os.environ["CUDA_VISIBLE_DEVICES"]) + 1) // 2)
    args.labels = ["normal", "tumor"]
    args.nb_per_class = [32*args.num_gpus, 32*args.num_gpus]
    args.batch_size = sum(args.nb_per_class)
    args.processed_size = [224, 224, 3]

    adda = ADDA(args)
    adda.define_source_encoder()
    
    if not args.train_discriminator:
        if args.eval_source_classifier is None:
            model = adda.get_source_classifier(adda.source_encoder, args.source_weights)
            adda.train_source_model(model, start_epoch=args.start_epoch-1) 
        else:
            model = adda.get_source_classifier(adda.source_encoder, args.eval_source_classifier)
            adda.eval_source_classifier(model, 'mnist')
            adda.eval_source_classifier(model, 'svhn')
    adda.define_target_encoder(args.source_weights)
    
    if args.train_discriminator:
        adda.train_target_discriminator(epochs=args.discriminator_epochs, 
                                        source_model=args.source_weights, 
                                        src_discriminator=args.source_discriminator_weights, 
                                        tgt_discriminator=args.target_discriminator_weights,
                                        start_epoch=args.start_epoch-1)
    if args.eval_target_classifier is not None:
        adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)
    
