'''
Helper functions for object detection
1. Create function for getting sliding windows within images
2. Create a readable pandas data frame for the labels.txt
3. Create a function for creating the input vector from images as a collection of row vectors
'''

import numpy as np
import pandas as pd
import os
from glob import glob

def sliding_window(image, xc, yc, size = 0.1, height = 30, width = 30):
    '''
    function to return a sliding window within each image 
    the function crops a portion of the image at the given coordinates with size = size
    it returns the cropped portion resized into height X width image
    '''
    x_max, y_max = image.size
    x1, x4 = (xc - size) * x_max, (xc + size) * x_max
    y1, y4 = (yc - size) * y_max, (yc + size) * y_max
    window = tuple(map(int, [x1, y1, x4, y4]))

    return window, image.crop(window).resize((width, height))

def get_input_vector(images):
    '''
    To create the input to a Machine Learning algorithm, we need to flatten the image windows and flatten each of them such that it is in format 
    N X Features, where N is the number of windows across all image windows and Features is the corresponding columns of each image
    '''

    return np.array([np.array(image).flatten() for image in images])

def get_phone_coordinates(df, file):
    '''
    We are provided labels.txt file containing the image name and corresponding phone center coordinates
    We convert the txt to a readable pandas data file
    We read the file name of the image and get its corresponding phone coordinates
    '''

    image_row = df[df['image'] == file]
    xc = float(image_row['xc'])
    yc = float(image_row['yc'])
    return xc, yc

def create_labels_table():
    '''
    Read labels.txt file and create a pandas dataframe / readable table from it that can be accessed easily
    '''
    folder = glob('data/find_phone/labels.txt')[0]
    table = pd.read_table(folder, sep=' ', names = ['image', 'xc', 'yc'])
    return table



