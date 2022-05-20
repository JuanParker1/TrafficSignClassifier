import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfile
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import os
from Dataset_tools import load_dataset_dict, image_to_ndarray
from Model import load_model
'''
TrafficSignClassifier.py


This is the main file that runs the application.

@Author:  Polo Lopez
'''


dataset_dir = os.getcwd() + '/TrafficSignClassifier/GTSRB'

dataset_dict = load_dataset_dict()
dataset_dict_reversed = dict([[v, k] for k, v in zip(dataset_dict.keys(), dataset_dict.values())])
model = load_model('regular_model')

def get_image_dir(subdir='Train', name=None, instance=None, resolution=5):
    directory = None
    if(subdir=='Test'):
        subdir = dataset_dir + '/Test/Final_Test/Images/'
        directory = subdir + (str(np.random.randint(len(os.listdir(subdir)))).zfill(5) if (instance==None) else str(instance).zfill(5)) + '.ppm'
    elif(subdir=='Train'):
        subdir = dataset_dir + '/Train/'
        subdir += (str(np.random.randint(len(os.listdir(subdir)))).zfill(5) if (name==None) else dataset_dict_reversed[name]) + '/'
        subdir += (np.random.choice(os.listdir(subdir)).split('_')[0] if (instance==None) else str(instance).zfill(5)) + '_'
        subdir += str(resolution + 24).zfill(5) + '.ppm'
        directory = subdir
    else:
        raise ValueError("The parameter 'subdir' must be either 'Train' or 'Test', but recieved:  '" + str(subdir) + "'")
    return directory

def predict_on_image(model, directory=None, subdir='Train', name=None, instance=None, resolution=5):
    if(not(directory)):
        directory = get_image_dir(subdir, name, instance, resolution)
    resolution = (model.layers[0].output_shape[0][1], model.layers[0].output_shape[0][2])  # infer resolution from model input layer
    grayscale =  model.layers[0].output_shape[0][-1] == 1                                  # infer grayscale from model input layer
    return model.predict(image_to_ndarray(directory, resolution, grayscale, for_prediction=True))


'''
Class that defines the application.
'''
class TSC_App:
    def __init__(self):
        # Root & canvas
        self.root = tk.Tk()
        self.root.title("Traffic Sign Classifier")
        self.canvas = tk.Canvas(self.root, width=650, height=650)
        self.canvas.grid(columnspan=4, rowspan=3)
        self.plot_figure = plt.figure(figsize=(6,3), dpi=100, tight_layout={'pad':1.5, 'h_pad':-17})

        self.image_dim = (300, 300)
        self.current_image_path = get_image_dir('Train', 'stop_sign', instance=8)  # default image
        self.current_gui_image = ImageTk.PhotoImage( Image.open(self.current_image_path).resize(self.image_dim) )

        # Image Widgit
        self.image_label = tk.Label(image=self.current_gui_image)
        self.image_label.Image = self.current_gui_image
        self.image_label.grid(columnspan=2, column=1, row=0)
        # Select Class Button
        self.select_class_button_selection = tk.StringVar()
        self.select_class_button_selection.set('stop_sign')
        self.select_class_button = tk.OptionMenu(self.root, self.select_class_button_selection, *dataset_dict_reversed.keys(), command=self.select_class_button_f)
        self.select_class_button.config(bg='#e0edea', height=3, width=16)
        self.select_class_button.grid(column=1, row=3)
        # Select From Dataset Button
        self.select_dataset_file_button = tk.Button(self.root, text="Select From Dataset", command=self.select_from_dataset_button_f, bg='#e0edea', fg='black', height=3, width=16)
        self.select_dataset_file_button.grid(column=2, row=3)
        # Plot the predictions
        self.__update_plot()

    def run(self):
        self.root.mainloop()
    
    # Update the current image display
    def __update_image_selection(self, directory):
        self.current_image_path = directory
        self.current_gui_image = ImageTk.PhotoImage( Image.open(self.current_image_path).resize(self.image_dim) )
        self.image_label.configure(image=self.current_gui_image)
        self.image_label.Image = self.current_gui_image
    
    # Update the prediction plot
    def __update_plot(self):
        predictions = predict_on_image(model, directory=self.current_image_path)
        sorted_indices = np.argsort(predictions)
        pred_sorted = predictions[0][sorted_indices]

        labels = np.array([ dataset_dict[ str(sorted_indices[0][-5:][i]).zfill(5) ] for i in range(5) ])
        values = pred_sorted[0][-5:]

        plt.clf()
        plt.subplot(2, 1, 2)
        plt.barh(labels, values)
        plt.xlabel('Confidence')
 
        plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.root)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().grid(columnspan=2, column=1, row=2)

        toolbar = NavigationToolbar2Tk(plot_canvas, self.root, pack_toolbar=False)
        toolbar.update()

    # Button functionality
    def select_class_button_f(self, selection):
        self.__update_image_selection(get_image_dir(name=selection))
        self.__update_plot()

    def select_from_dataset_button_f(self):
        im_file = askopenfile(parent=self.root, initialdir=dataset_dir, mode='r', title="Select an Image File", filetype=[("ppm file", '*.ppm')])
        self.__update_image_selection(im_file.name)
        self.__update_plot()

TSC_App().run()