# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:30:34 2021

@author: jordy
"""

# Algorithm for Image Array Conversion

# -- [Part One: Save Plots as Images] --

# Step 1) Import Necessary Libraries
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

import sys

# Step 2) Load .csv File
df = pd.read_csv('../Burman_J_2017 Station Count.csv')

# Step 3) Create a Plot from One Row
fig = plt.figure()
plot   = fig.add_subplot (111)
# plt.clf()
plt.axis('off')
plot.plot(df.drop(['Holiday','Station Name'],axis=1).iloc[0,:])
plt.savefig("tmp.png")
image = Image.open("tmp.png")
np_im = np.array(image) 
# print(np_im.shape)

# plt.imshow(np_im[48:288-48,60:432-60,:])
newim=np_im[38:288-38,50:432-50,:]
# plt.savefig('plot.png', dpi=300, bbox_inches='tight')
im = Image.fromarray(newim)
im.save("plot.png")

# Step 4) Save Plot Figure from One Row as Image
#plt.savefig("graph0.png")
# st = 0

# Step 5) Create For-Loop that will Create Plot and Save Plot as Image
for i in range(0, 4015):
    fig = plt.figure()
    plot = fig.add_subplot (111)
    plt.axis('off')
    plot.plot(df.drop(['Holiday','Station Name'],axis=1).iloc[i,:])
    plt.savefig("tmp.png")
    image = Image.open("tmp.png")
    np_im = np.array(image) 
    # print(np_im.shape)
    newim=np_im[38:288-38,55:432-55,:]
    im = Image.fromarray(newim)
    
    title = "graph" + str(i) + ".png"
    im.save(title)
    # plt.savefig(title, dpi=300, bbox_inches='tight')
    # plt.savefig(title)
    if(i % 200 == 0 or i == 4015):
       print(str(i) + " out of 4015 complete")

sys.exit()
# -- [Part One Complete] --

# -- [Part Two: Opening Images and Gathering Color Sequence Data]--

# Step 6) Using Pillow, Open One Image
imageOne = Image.open("graph0.png") 

# Step 7) Create a Variable, and Store Image Data into Variable
image_sequence = image.getdata()


# Step 8) Convert Variable into Numpy Array and Store it into New Variable (MAKE SURE TO ROLL THE ALPHA CHANNEL TO THE FRONT)
image_array = np.array(image_sequence).reshape(1,-1)
sys.exit()

# Step 9) Create For-Loop that will Open Images, Gather Data, and Store Data into Matrix (MAKE SURE TO RESHAPE AND USE VSTACK!)
combined_array = np.empty([1,497664], dtype = np.int32)
for i in range(4000, 4015):
    title = "graph" + str(i) + ".png"
    img_array = np.array(Image.open(title).getdata()).reshape(1,-1)
    combined_array = np.vstack([combined_array, img_array])
    if(i % 200 == 0 or i == 4014):
        print(str(i) + " out of 4014 complete")
    
DF = pd.DataFrame(combined_array)
DF.to_csv("imgarr_5.csv")
# Step 10) Once Matrix is Complete, Save Matrix as .csv File

# -- [Part Two Complete] --
