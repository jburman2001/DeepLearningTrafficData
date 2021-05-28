# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 01:18:55 2021

@author: jordy
"""

# Step 1) Import Necessary Libraries

# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from skimage import io, transform

# PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Eps parameter
eps = 50
lrt = 0.001

# Load .csv File
# df = pd.read_csv('Burman_J_2017 Station Count.csv')


#list=[]
#for i in range(0,4015):
  # list.append("graph" + str(i) + ".png")
#imlist=pd.DataFrame(list)
#imlist.columns=['file_name']
#imlist['label']=df['Holiday']
#imlist
# imlist.to_csv('imlist.csv',index=False)

# load & split csv file


#imlist=pd.read_csv('imlist.csv')



# training set
#train = imlist.sample(frac = 0.8)
#train.to_csv('train.csv',index=False)



# test set
#test = imlist.drop(train.index)
#test.to_csv('test.csv',index=False)

# START HERE




class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.landmarks_frame = pd.read_csv(csv_file)


    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        img_loc = self.landmarks_frame.iloc[idx,0]

        # image = torch.tensor(io.imread(img_loc)) change 4 channels to 3, change unit to float
        image = Image.open(img_loc)
        image.thumbnail((212,322))
        image = image.convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255
        image = image[:, :, :3]
        image = torch.tensor(image)
    
        label = self.landmarks_frame.iloc[idx, 1].astype('float')
    
        sample = {'image': image, 'label': label}


        return sample


# Create the train set below:
train_dataset = CustomDataSet(csv_file='train.csv')



#plt.imshow(train_dataset[60]['image'])
#print(train_dataset[60]['label'],len(train_dataset))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)

# Create the test set below:
test_dataset = CustomDataSet(csv_file='test.csv')



# test
#plt.imshow(test_dataset[9]['image'])
#print(test_dataset[9]['label'],len(test_dataset))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# next, define NN and start training
# class Net(nn.Module): ...
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 32 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
    def forward(self, x):
        #print(x.size())
        x = np.transpose(x,(0,3,1,2))
        #print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = x.view(-1, 16 * 32 * 50)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.size())
        return x


net = Net()

# Loss
criterion = nn.CrossEntropyLoss()
# lr impacts Performance of Neural Network
optimizer = optim.SGD(net.parameters(), lr = lrt, momentum=0.9)

# Training
for epoch in range(eps):  # loop over the dataset x number of times
    print(str(epoch) + " of " + str(eps) + " complete.")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Summaries

# Testloader
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data['image'],data['label']
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
# print(predicted.shape)
        correct += (predicted == labels.long()).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Trainloader
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data['image'],data['label']
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
# print(predicted.shape)
        correct += (predicted == labels.long()).sum().item()
print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

# Save
PATH = './Burman_J_Holiday_net.pth'
torch.save(net.state_dict(), PATH)
