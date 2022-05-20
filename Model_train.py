from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
from Dataset_tools import load_dataset
from Model import regular_classifier
'''
Model_train.py


Code for training and logging the model.

Steps for running Tensorboard:
    (With Tensorboard extension in VS Code):
    1)  Ctrl + Shift + P  to view commands
    2)  Select "Launch Tensorboard" >> "Use current working directory" (or select folder where logs is contained)
    NOTE:  Refresh a couple of times until the graphs show up

    (Without Tensorboard extension):
    1)  Open a separate terminal and run "tensorboard --logdir=~/TrafficSignClassifier/logs"
    2)  Go to the link provided in your web browser (Ex:  "http://localhost:6006/")

@Author:  Polo Lopez
'''


dataset_dir = os.getcwd() + '/TrafficSignClassifier/GTSRB'
logging_dir = os.getcwd() + "/TrafficSignClassifier/logs"
model_save_dir = os.getcwd() + "/TrafficSignClassifier/saved_models"

model_name = 'new_model'
num_epochs = 50
batch_size = 16

# Basic parameters for image data
im_resolution = (32, 32)
grayscale = True

# The model that is used for traning
model = regular_classifier(resolution=im_resolution, grayscale=grayscale)

train_set, valid_set = load_dataset(dataset_dir, im_resolution, grayscale=grayscale)     #  (images, labels)

# Model training function
def train_model(model, epochs=10, batch_size=32, patience=5, name='model'):
    # image transformation data generator
    datagen = ImageDataGenerator(fill_mode='constant', rotation_range=4, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, zoom_range=0.15)
    train_generator = datagen.flow(train_set[0], train_set[1], batch_size=batch_size)
    valid_generator = datagen.flow(valid_set[0], valid_set[1], batch_size=batch_size)
    
    # callbacks for model saving and logging
    checkpoint_cb = ModelCheckpoint(model_save_dir + '/' + name, save_best_only=True)
    tensorboard_cb = TensorBoard(log_dir=logging_dir, histogram_freq=1)
    early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights=True)
    # run training
    history = model.fit_generator(train_generator, validation_data=valid_generator, epochs=epochs, callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb])
    return history

# Logging Function  (produces summary.txt file with overview of model training (inside the '/logs' directory))
def make_log(history):
    log_file = open(logging_dir + '/summary.txt', mode='w')
    log_file.write("epoch:\ttraining_loss:\tvalidation_loss:\n")
    for e in range(len(history.history['loss'])):
        log_file.write(str(e) + '\t' + str(history.history['loss'][e]) + '\t' + str(history.history['val_loss'][e]) + '\n')
    log_file.close()

# Training and logging
history = train_model(model, epochs=num_epochs, batch_size=batch_size, name=model_name)
make_log(history)