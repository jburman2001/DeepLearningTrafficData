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

# Parameters
eps = 1000
lrt = 0.001
PATH = './Burman_J_Holiday_net_'+str(eps)+str(lrt)+'.pth'
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
#train = imlist.sample(frac = 0.85)
# handle the unbalanced datasets
#samp_rate=min(len(train[train['label']==1]),len(train[train['label']==0]))/max(len(train[train['label']==1]),len(train[train['label']==0]))
#if len(train[train['label']==1])>len(train[train['label']==0]):
    #to_drop=train[train['label']==1].sample(frac = 1-samp_rate)
#else:
    #to_drop=train[train['label']==0].sample(frac = 1-samp_rate)
#df4 = train.drop(to_drop.index)
# df4
#df4.to_csv('train50.csv',index=False)



# test set
#test=pd.read_csv('test.csv')
#test.to_csv('test.csv',index=False)
# handle the unbalanced datasets
#samp_rate=min(len(test[test['label']==1]),len(test[test['label']==0]))/max(len(test[test['label']==1]),len(test[test['label']==0]))
#if len(test[test['label']==1])>len(test[test['label']==0]):
   #to_drop=test[test['label']==1].sample(frac = 1-samp_rate)
#else:
    #to_drop=test[test['label']==0].sample(frac = 1-samp_rate)
#df4 = test.drop(to_drop.index)
# df4
#df4.to_csv('test50.csv',index=False)



# START HERE


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()

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
# If you want to use the entire (unbalanced) dataset use this:
#train_dataset = CustomDataSet(csv_file='train.csv')
# If you want to use a balanced set use this:
train_dataset = CustomDataSet(csv_file='train50.csv')

#plt.imshow(train_dataset[60]['image'])
#print(train_dataset[60]['label'],len(train_dataset))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)

# Create the test set below:
# If you want to use the entire (unbalanced) dataset use this:
#test_dataset = CustomDataSet(csv_file='test.csv')
# If you want to use a balanced set use this:
test_dataset = CustomDataSet(csv_file='test50.csv')


# test
#plt.imshow(test_dataset[9]['image'])
#print(test_dataset[9]['label'],len(test_dataset))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=0)


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
lossV = []
for epoch in range(eps):  # loop over the dataset x number of times
    if(epoch % 5 == 0 or epoch == 99):    
        print(str(epoch) + " of " + str(eps - 1) + " complete.")
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
        #running_loss += loss.item()
        #if i % 2000 == 1999:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
           #running_loss = 0.0
    lossV.append(loss.item())


print('Finished Training')
print(lossV)
plt.plot(lossV)
# Summaries

# Testloader
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data['image'], data['label']
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        #plogic = (predicted==labels.long())&(labels.long()!=0)
        #print(plogic)
        #ttpic = inputs[(predicted == labels.long())]
        #ttpic = inputs[plogic]
        # The Predicted Value
        #mlvalue = predicted[plogic]
        # The Actual Label Value
        #mlabel = labels[plogic]
        #imshow(torchvision.utils.make_grid(ttpic[3]))
        #plt.imshow(ttpic[3])
        #plt.show()
        #print(mlvalue[3])
        #print(mlabel[3])
        total += labels.size(0)
# print(predicted.shape)
        correct += (predicted == labels.long()).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Trainloader
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data['image'], data['label']
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
# print(predicted.shape)
        correct += (predicted == labels.long()).sum().item()
print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

# Save

torch.save(net.state_dict(), PATH)
