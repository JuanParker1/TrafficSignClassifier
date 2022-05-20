import numpy as np
from PIL import Image
import os
from tensorflow.keras.utils import normalize
'''
Dataset_tools.py


Contains several functions for loading the dataset, as well as functions for
processing and converting image data.

@Author:  Polo Lopez
'''


# Function for loading the dataset
def load_dataset(directory, resolution=(32, 32), grayscale=False, train_test_split=0.8, normalize_data=True):
    x, y = [], []
    num_classes = 43
    for folder_name in os.listdir(directory + '/Train/'):
        for ppm_image in os.listdir(directory + '/Train/' + folder_name + '/'):
            x.append(image_to_ndarray(directory + '/Train/' + folder_name + '/' + ppm_image, resolution, grayscale))
            one_hot_label = np.zeros(num_classes)
            one_hot_label[int(folder_name)] = 1
            y.append(one_hot_label)
    x, y = np.array(x), np.array(y)
    if(normalize_data):  x = normalize(x)
    if(grayscale):  x = np.reshape(x, (*x.shape, 1))

    # Shuffle the data
    rand_indices = np.arange(len(x))
    np.random.shuffle(rand_indices)
    x, y = [x[i] for i in rand_indices], [y[i] for i in rand_indices]
    x, y = np.array(x), np.array(y)

    # Split into training and validation sets
    index_bound = int(len(x)*train_test_split)
    train_set = (x[:index_bound], y[:index_bound])
    valid_set = (x[index_bound:], y[index_bound:])

    return train_set, valid_set

def load_dataset_dict():
    path = os.getcwd() + '/TrafficSignClassifier/GTSRB/dataset_dict.txt'
    dataset_dict = dict()
    file = open(path, 'r')
    for line in file.readlines():
        line_split = line.split(': ')
        dataset_dict[line_split[0]] = line_split[1].replace('\n', '')
    file.close()
    return dataset_dict

# translates an image file to numpy array
def image_to_ndarray(directory, resolution=(256, 256), grayscale=False, for_prediction=False):
    if(grayscale):
        image = Image.open(directory).convert('L')
    else:
        image = Image.open(directory)
    image = image.resize((resolution[0], resolution[1]))
    array = np.array(image, dtype='float32') / 256
    if(for_prediction):  array = array.reshape(1, *array.shape, 1)  if  (grayscale)  else  array.reshape(1, *array.shape)
    return(array)

# resizes numpy-array-stored image to specified dimensions (default: 256x256)
def ndarray_resized(array, resolution=(256, 256)):
    new_array = np.array(array*256, dtype='uint8')
    image = Image.fromarray(new_array)
    image.resize((resolution[0], resolution[1]))
    new_array = np.array(image, dtype='float32') / 256
    return(new_array)

# displays image in a seperate window
def show_image(directory, resolution=(256, 256)):
    image = Image.open(directory)
    image = image.resize((resolution[0], resolution[1]))
    image.show()