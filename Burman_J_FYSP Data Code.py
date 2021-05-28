import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# load csv
df=pd.read_csv('Burman_J_2017 Station Count.csv')

# generate plot from one row
#fig = plt.figure()
#plot   = fig.add_subplot (111)
#plot.plot(df.drop(['Holiday','Station Name'],axis=1).iloc[0,:])
#plt.show()

#fig.canvas.draw()

#w,h = fig.canvas.get_width_height()
#buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#buffer.shape = (w,h , 4)

#buffer = np.roll(buffer, 3, axis = 2)


# function to convert plot to numpy array
#def fig2data ( fig ):

    
    # draw the renderer
    #fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
   # w,h = fig.canvas.get_width_height()
    #buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
   # buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
   # return buf

# convert a plot to img array
#img=fig2data(figure)

# New Function (PLEASE REVIEW THIS)
def plotdata(df0, a):
    fig = plt.figure()
    plot   = fig.add_subplot (111)
    plot.plot(df.drop(['Holiday','Station Name'],axis=1).iloc[a,:])
    
    
    
    fig.canvas.draw()

    w,h = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buffer.shape = (w,h , 4)

    buffer = np.roll(buffer, 3, axis = 2)
    
    return buffer
    



# df.apply()
# df.apply(plotdata(df, ), axis = 1)

# for loop

test = plotdata(df, 1)

plt.imshow(test)

#an_image = Image.open("graph0.png") # open the image.
#image_sequence = an_image. getdata()
#image_array = np. array(image_sequence)

#print(image_array)

# combined_array = np.array([], dtype=np.int64)
#st = 0
#data = plotdata(df, st)
#print(data.shape)
#combined_array = data.reshape(1,-1)
# test = plotdata(df, 0)
# test = test.reshape(-1,1)

# combined_array = None
#for i in range(st + 1, st + 1000):
   # data = plotdata(df, i)
   # combined_array = np.vstack([combined_array, data])
   # if(i % 200 == 0 or i == 1000):
       # print(str(i) + " out of 100 complete")
        
# Check Shape of Array 
#print(combined_array.shape)
# Save Result
#DF = pd.DataFrame(combined_array)
#print(len(DF))
#img_file = "img_array_" + str(st) + ".csv"
#print(img_file)
#DF.to_csv(img_file)

