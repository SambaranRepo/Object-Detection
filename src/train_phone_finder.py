'''
Code to run the training of the phone finder model 
The architecture of the code is to first create windows in each image
One window contains the phone which can be generated using the phone coordinates provided in labels.txt
Another window from same image that does not contain the image. This is created by using random xc and yc which is not equal to the phone coordinates
All such windows are concatenated to form the input vector with labels as true if it contains phone and false if it does not contain the phone
We can fit a linear regression model that takes in the window input features and the ground truth label and train the classifier
'''

import numpy as np
import os, sys
from PIL import Image
import random
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from utils import *

def create_training_validation_split(folder, train_val_split = 0.8):
    '''
    Divide the given images into training and validation set
    we use a 80% split
    '''

    files = sorted(list(os.listdir(folder)))
    N = len(files)
    print(f"number of files : {N}")
    training_set_length = int(train_val_split * N)
    training_files = files[:training_set_length]
    validation_files = files[training_set_length:]
    return training_files, validation_files

def create_labeled_dataset(files, window_size = 0.05):
    '''
    From each image, create a positive example window and a negative example window as explained in code beginning
    '''
    folder = glob('data/find_phone/')[0]
    df = create_labels_table()
    dataset = []
    for file in files:
        try:
            image = Image.open(folder + file)
            file_name = os.path.basename(os.path.normpath(file))
            
            #Create window containing phone
            xc, yc = get_phone_coordinates(df, file_name)
            _,window = sliding_window(image, xc, yc, window_size)
            dataset += [(window,True)]

            #Create negative window
            _, window = sliding_window(image, random.random(), random.random(), window_size)
            dataset += [(window, False)]
        except Exception as e:
            pass

    return dataset

def create_training_validation_dataset(folder, window_size = 0.05):
    '''
    Create the training dataset and testing dataset using the window and its corresponding label from the function above
    '''
    training_files, validation_files = create_training_validation_split(folder)
    training_dataset = create_labeled_dataset(training_files)
    validation_dataset = create_labeled_dataset(validation_files)
    return training_dataset, validation_dataset

def train(x_train, y_train):
    '''
    Train the logistic regression model on the training dataset
    '''
    classifier = LogisticRegression(max_iter = 20000)
    classifier.fit(x_train, y_train)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
def train_model(folder):
    '''
    Main training loop set up on the training data
    '''
    training_dataset, validation_dataset = create_training_validation_dataset(folder)
    print(training_dataset)
    training_images, training_labels = zip(*training_dataset)
    validation_images, validation_labels = zip(*validation_dataset)

    x_train, x_val = get_input_vector(training_images), get_input_vector(validation_images)

    train(x_train, training_labels)



if __name__ == '__main__':
    train_model(sys.argv[1])

