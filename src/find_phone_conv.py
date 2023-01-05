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
from train_phone_finder_conv import CustomNetwork
import torch

def create_window_coordinates(window_size = 0.1, num_windows = 50):
    '''
    Get the center normalized coordinates of each window in the image
    We want to create 40 windows in each image across each direction
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

def predict(image_file, show_image = False, window_size = 0.1):
    '''
    Prediction loop on a given image
    '''
    folder = glob('data/find_phone/')[0]
    image = Image.open(folder + image_file)
    xc, yc, boxes, windows = zip(*create_sliding_windows(image, window_size))
    windows = get_input_vector(windows)/255
    windows = torch.tensor(windows, dtype = torch.float32)
    windows = windows.reshape([-1, 1, 2700])
    model = CustomNetwork()
    # model.load_state_dict(torch.load('conv_net_1.pt'))
    model.load_state_dict(torch.load('models/conv_net_1.pt'))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model.to(device)
    with torch.no_grad():
        probs = (model.forward(windows.to(device))[:,1])
        max_prob_window = np.argmax(probs.cpu().numpy())
        if show_image:
            draw = ImageDraw.Draw(image)
            box = boxes[max_prob_window]
            # print(box)
            draw.rectangle(box, outline= 'red')
            plt.imshow(np.array(image))
            # plt.show(block = True)
            plt.savefig(f'results/images/{image_file}')
            plt.close()

    return xc[max_prob_window], yc[max_prob_window]

def accuracy(window_size = 0.05):
    '''
    '''
    folder = glob('data/find_phone/')[0]
    files = sorted(os.listdir(folder))
    df = create_labels_table()
    num_correct = 0
    for file in tqdm(files):
        try:
            image_row = df[df['image'] == file]
            x_gt, y_gt = image_row['xc'], image_row['yc']
            x_pred, y_pred = predict(file, show_image= True, window_size=window_size)
            error = np.linalg.norm([x_gt - x_pred, y_gt - y_pred])
            if error < 0.05:
                num_correct += 1
        except Exception as e:
            pass
    print("Accuracy of algorithm on the dataset is : {:02f}%".format(100 * num_correct / len(files)))

if __name__ == '__main__':
    predict_x, predict_y = predict(sys.argv[1], True, window_size = 0.08)
    print(predict_x, predict_y)
    accuracy()
