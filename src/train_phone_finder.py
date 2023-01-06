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
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from sklearn.linear_model import LogisticRegression
from utils import *
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument(method, help='Specify the method to use : 1. For SKLEARN LogisticRegrression \n 2. For Custom Deep Learning network', nargs='?', type=int, const=1, default=2)
parser.add_argument('folder', help='Location to dataset', nargs='?', const=1, type=str, default='./data/find_phone')
parser.add_argument('method', help='Specify the model to use : 1. LogisticRegression model \n 2. Deep Learning model', nargs='?', const=1, type=int, default=2)
args = parser.parse_args()
folder = args.folder
method = args.method
assert method in [1,2], 'Please enter method to be either 1 or 2, 1 for SKLEARN LogisticRegression and 2 for Deep Learning Categorical Classification'

class CustomNetwork(nn.Module):
    '''
    : Create a deep learning network
    : 2 convolutional layers + 3 fully connected layers
    : Input is sliding windows within the image
    : Output is for each window a (0,1) output with 0 indicating that window does not contain phone, 1 indicating window contains phones
    '''

    def __init__(self, num_features = 2700):
        super().__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(1, 16, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv1d(16, 32 ,5, stride = 1, padding = 2)
        self.conv3 = nn.Conv1d(32, 32 ,5, stride = 1, padding = 2)
        self.fc1 = nn.Linear(self.num_features  * 32, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8,2)
        self.relu = nn.ReLU()

        self.sequential = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)

    def forward(self, x):
        x = self.sequential(x)
        x = x.view([-1, 32 * self.num_features])
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.relu((self.fc3(x)))
        return x


def create_training_validation_split(folder, train_val_split = 0.8):
    '''
    Divide the given images into training and validation set
    we use a 80% split
    '''

    files = np.array(sorted(list(os.listdir(folder))))
    N = len(files)
    print(f"number of files : {N}")
    training_set_length = int(train_val_split * N)
    training_files = files[:training_set_length]
    validation_files = files[training_set_length:]
    training_files = [folder + '/' + file for file in training_files]
    validation_files = [folder + '/' + file for file in validation_files]
    return training_files, validation_files

def create_labeled_dataset(files, window_size = 0.1):
    '''
    From each image, create a positive example window and a negative example window as explained in code beginning
    '''
    df = create_labels_table(folder)
    dataset = []
    for file in files:
        try:
            image = Image.open(file)
            file_name = os.path.basename(os.path.normpath(file))
            #Create window containing phone
            xc, yc = get_phone_coordinates(df, file_name)
            _,window = sliding_window(image, xc, yc, window_size)
            dataset += [(window,1)]

            #Create negative window
            _, window = sliding_window(image, random.random(), random.random(), window_size)
            dataset += [(window, 0)]
        except Exception as e:
            pass

    return dataset

def create_training_validation_dataset(folder, window_size = 0.1):
    '''
    Create the training dataset and testing dataset using the window and its corresponding label from the function above
    '''
    training_files, validation_files = create_training_validation_split(folder)
    training_dataset = create_labeled_dataset(training_files, window_size= window_size)
    validation_dataset = create_labeled_dataset(validation_files, window_size= window_size)
    return training_dataset, validation_dataset

def train_regression(x_train, y_train):
    '''
    Train the logistic regression model on the training dataset
    '''
    classifier = LogisticRegression(max_iter = 20000)
    classifier.fit(x_train, y_train)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

def train_conv_net(model, optimizer, training_images, training_labels, validation_images, validation_labels, loss_criteria = nn.CrossEntropyLoss(), lr = 3e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    training_labels = torch.tensor(training_labels, dtype = torch.long)
    validation_labels = torch.tensor(validation_labels, dtype = torch.long)
    train_loss = []
    val_it = []
    val_loss = []
    for it in tqdm(range(10000)):
        if (it % 1000 == 0):
            new_lr = lr * (0.5**(it / 2000))
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = new_lr
        model.train()
        output = model.forward(training_images.to(device))
        # print(output)
        loss = loss_criteria(output, training_labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print("Epoch : {} =====> Loss : {:02f}".format(it + 1, loss.item()))

        if (it % 25 == 0):
            model.eval()
            with torch.no_grad():
                output = model.forward(validation_images.to(device))
                loss = loss_criteria(output, validation_labels.to(device))
                val_loss.append(loss.item())
                val_it.append(it)
    
    fig, ax = plt.subplots(1,2,figsize=(10,8))
    ax[0].plot(range(it + 1), train_loss)
    ax[0].set_title('Training loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Training loss')
    ax[1].plot(val_it, val_loss)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Validations loss')
    ax[1].set_title('Validation Loss')
    plt.savefig('results/train_val.png', format = 'png', bbox_inches = 'tight')
    plt.show(block = True)
    plt.close()

    torch.save(model.state_dict(),  'models/conv_net_3.pt' )



def train_sklearn(folder, window_size = 0.05):
    training_dataset, validation_dataset = create_training_validation_dataset(folder, window_size= window_size)
    training_images, training_labels = zip(*training_dataset)
    validation_images, validation_labels = zip(*validation_dataset)    
    x_train, x_val = get_input_vector(training_images) / 255, get_input_vector(validation_images) / 255
    train_regression(x_train, training_labels)


def train_conv(folder, window_size = 0.05):
    '''
    Main training loop set up on the training data
    '''
    training_dataset, validation_dataset = create_training_validation_dataset(folder, window_size= window_size)
    training_images, training_labels = zip(*training_dataset)
    validation_images, validation_labels = zip(*validation_dataset)
    
    x_train, x_val = get_input_vector(training_images) / 255, get_input_vector(validation_images) / 255
    x_train = torch.tensor(x_train , dtype = torch.float32)
    x_val = torch.tensor(x_val, dtype = torch.float32)
    num_features = x_train.shape[1]
    x_train = x_train.reshape([-1, 1, num_features])
    x_val = x_val.reshape([-1, 1, num_features])
    model = CustomNetwork(num_features=num_features)
    lr = 3e-3
    optimizer = SGD(params = model.parameters(), lr = lr)

    train_conv_net(model = model, optimizer= optimizer, training_images= x_train, training_labels= training_labels, validation_images= x_val, validation_labels= validation_labels, lr = lr)



if __name__ == '__main__':
    if method == 1:
        train_sklearn(folder, window_size = 0.05)
    else:
    	train_conv(folder, window_size = 0.05)

