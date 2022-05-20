import os
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
'''
Model.py


This file includes the model definition and function for loading the model.

@Author:  Polo Lopez
'''


model_save_dir = os.getcwd() + "/TrafficSignClassifier/saved_models"

def load_model(name):
    return tf.keras.models.load_model(model_save_dir + '/' + name)

def regular_classifier(resolution=(32, 32), grayscale=False):
    image_placeholder = Input(shape=(resolution[0], resolution[1], 1 if grayscale else 3))
    x = Convolution2D(filters=32, kernel_size=(5, 5), activation='relu') (image_placeholder)
    x = MaxPooling2D(pool_size=(2, 2)) (x)
    pool1 = MaxPooling2D(pool_size=(2, 2)) (x)
    x = Dropout(0.2) (x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), activation='relu') (x)
    pool2 = MaxPooling2D(pool_size=(2, 2)) (x)
    concatenated_pools = Concatenate() ([Flatten() (pool1), Flatten() (pool2)])
    x = Dropout(0.35) (concatenated_pools)
    x = Dense(512, activation='relu') (x)
    x = Dropout(0.4) (x)

    output_layer = Dense(43, activation='softmax') (x)
    model = Model(inputs=[image_placeholder], outputs=[output_layer])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')
    return model