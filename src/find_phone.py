'''
This code is used to run test on the images using the trained model
Here we now need to divide the images into windows, and for each window see whether it contains phone or not using the classifier predict method
For each window, classifier will give 2 outputs : the boolean of whether window contains phone or not, and the probability / confidence
We select the window with maximum probability

The argmax function gives us the appropriate window using which we can get the normalized coordinates of the window / center
'''

import numpy as np
import os, sys
from utils import get_input_vector, sliding_window, create_labels_table
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
from glob import glob
from tqdm import tqdm
from train_phone_finder import CustomNetwork
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file', help='Location to dataset', nargs='?', const=1,type=str)
parser.add_argument('method', help='Specify the model to use : 1. LogisticRegression model \n 2. Deep Learning model', nargs='?', const=1, type=int, default=2)
args = parser.parse_args()
file = args.file
method = args.method
assert method in [1,2], 'Please enter method to be either 1 or 2, 1 for SKLEARN LogisticRegression and 2 for Deep Learning Categorical Classification'

method_dict = {1 : 'LogisticRegression', 2 : 'ConvNet'}

def create_window_coordinates(window_size = 0.1, num_windows = 50):
    '''
    Get the center normalized coordinates of each window in the image
    We want to create 50 windows in each image across each direction
    '''

    coords = np.linspace(window_size, 1 - window_size, num_windows)
    return coords

def create_sliding_windows(image, window_size = 0.1, num_windows = 50):
    '''
    Create the sliding windows using the sliding window coordinates
    '''
    xc, yc = create_window_coordinates(window_size), create_window_coordinates(window_size)
    for x in xc:
        for y in yc:
            box, window = sliding_window(image, x, y, window_size)
            yield x,y,box,window

def predict(image_file, show_image = False, show_individual_result = False, window_size = 0.1):
    '''
    : Prediction loop on a given image
    : Prints the normalized phone center coordinates. 
    : If using LogisitcRegression, images with phone bounding box are saved in results/images_LogisticRegression folder and the coordinates are saved in results/labels_LogisticRegression.txt
    : If using the custom deep learning network, images with bounding box are saved in results/images_ConvNet folder and coordinates are saved in results/labels_ConvNet.txt
    : Main idea of the prediction is to divide the image into windows, run the trained model on each windows and select the window with max probability of containing phone
    : Return the normalized phone coordinates of the max phone probability window. 
    '''
    image = Image.open(image_file)
    filename = image_file.split('/')[-1]
    xc, yc, boxes, windows = zip(*create_sliding_windows(image, window_size))
    windows = get_input_vector(windows)/255

    if method == 1:
        with open('models/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        predicted_probs = classifier.predict_proba(get_input_vector(windows))[:,1]
        max_prob_window = np.argmax(predicted_probs)

    elif method == 2:
        windows = torch.tensor(windows, dtype = torch.float32)
        windows = windows.reshape([-1, 1, windows.shape[1]])
        model = CustomNetwork(num_features = windows.shape[-1])
        model.load_state_dict(torch.load('models/conv_net_1.pt'))
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        with torch.no_grad():
            probs = (model.forward(windows.to(device))[:,1])
            max_prob_window = np.argmax(probs.cpu().numpy())
    
    if show_image:
        draw = ImageDraw.Draw(image)
        box = boxes[max_prob_window]
        draw.rectangle(box, outline= 'red')
        plt.imshow(np.array(image))
        if show_individual_result:
            plt.show(block = True)
        plt.savefig(f'results/images_{method_dict[method]}/{filename}')
        plt.close()

    return xc[max_prob_window], yc[max_prob_window]

def accuracy(window_size = 0.05):
    '''
    : Compute the normalized phone coordinates in each image, and compare with the ground truth results given in data/find_phone/labels.txt
    : Success if predicted coordinates are within a circle of radius 0.05 of the ground truth coordinates
    : Print the overall accuracy of the algorithm for the provided dataset 
    '''
    folder = glob('data/find_phone/')[0]
    files = sorted(os.listdir(folder))
    df = create_labels_table()
    num_correct = 0
    with open(f'results/labels_{method_dict[method]}.txt', 'w') as f:
        for file in tqdm(files):
            try:
                image_row = df[df['image'] == file]
                x_gt, y_gt = image_row['xc'], image_row['yc']
                x_pred, y_pred = predict(folder + file, show_image= True, window_size=window_size)
                f.write(file + ' ' + str(x_pred) + ' ' + str(y_pred) + '\n')
                error = np.linalg.norm([x_gt - x_pred, y_gt - y_pred])
                if error < 0.05:
                    num_correct += 1
            except Exception as e:
                pass
        print("Accuracy of algorithm on the dataset is : {:02f}%".format(100 * num_correct / len(files)))

if __name__ == '__main__':
    predict_x, predict_y = predict(file, True, window_size = 0.05)
    print(predict_x, predict_y)
    accuracy(window_size=0.05)
