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
from utils import *
import matplotlib.pyplot as plt

class CustomNetwork(nn.Module):
    def __init__(self, output_dim = None):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv1d(16, 32 ,5, stride = 1, padding = 2)
        self.conv3 = nn.Conv1d(32, 32 ,5, stride = 1, padding = 2)
        self.fc1 = nn.Linear(2700  * 32, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8,2)
        self.relu = nn.ReLU()

        # self.sequential = nn.Sequential(self.conv1, self.relu, self.max_pool, self.conv2, self.relu, self.max_pool)
        self.sequential = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)

    def forward(self, x):
        x = self.sequential(x)
        # print(f'shape of x after conv : {x.shape}')
        x = x.view([-1, 32 * 2700])
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.relu((self.fc3(x)))
        return x
    
def conv_train(model, optimizer, training_images, training_labels, validation_images, validation_labels, loss_criteria = nn.CrossEntropyLoss()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    training_labels = torch.tensor(training_labels, dtype = torch.long)
    validation_labels = torch.tensor(validation_labels, dtype = torch.long)
    train_loss = []
    val_it = []
    val_loss = []
    for it in tqdm(range(10000)):
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
    ax[1].plot(val_it, val_loss)
    ax[1].set_title('Validation Loss')
    plt.show(block = True)
    plt.close()

    torch.save(model.state_dict(),  'conv_net_1.pt' )



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

def create_labeled_dataset(files, window_size = 0.1):
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

def train(x_train, y_train):
    '''
    Train the logistic regression model on the training dataset
    '''
    classifier = LogisticRegression(max_iter = 20000)
    classifier.fit(x_train, y_train)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
def train_model(folder, window_size = 0.05):
    '''
    Main training loop set up on the training data
    '''
    training_dataset, validation_dataset = create_training_validation_dataset(folder, window_size= window_size)
    training_images, training_labels = zip(*training_dataset)
    validation_images, validation_labels = zip(*validation_dataset)
    
    x_train, x_val = get_input_vector(training_images) / 255, get_input_vector(validation_images) / 255
    x_train = torch.tensor(x_train , dtype = torch.float32)
    x_val = torch.tensor(x_val, dtype = torch.float32)
    x_train = x_train.reshape([-1, 1, 2700])
    x_val = x_val.reshape([-1, 1, 2700])
    model = CustomNetwork()
    optimizer = SGD(params = model.parameters(), lr = 1e-3)

    conv_train(model = model, optimizer= optimizer, training_images= x_train, training_labels= training_labels, validation_images= x_val, validation_labels= validation_labels)



if __name__ == '__main__':
    train_model(sys.argv[1], window_size = 0.05)

