import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt



# load csv
df=pd.read_csv('Burman_J_2017 Station Count.csv')

# generate plot from one row
figure = mpl.pyplot.figure(  )
plot   = figure.add_subplot ( 111 )
plot.plot(df.drop(['Holiday','Station Name'],axis=1).iloc[0,:])
mpl.pyplot.show()

# function to convert plot to numpy array
def fig2data ( fig ):

    
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

# convert a plot to img array
img=fig2data(figure)

# New Function (PLEASE REVIEW THIS)
def plotdata(df0, a):
    figure = mpl.pyplot.figure()
    plot   = figure.add_subplot ( 111 )
    plot.plot(df0.drop(['Holiday','Station Name'],axis=1).iloc[a,:])
        
    # draw the renderer
    figure.canvas.draw ( )
     
    # Get the RGBA buffer from the figure
    w,h = figure.canvas.get_width_height()
    buf = np.fromstring ( figure.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
     
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )

    return buf


# df.apply()
# df.apply(plotdata(df, ), axis = 1)

# for loop

empty = np.array([], dtype=np.int64)
test = plotdata(df, 0)
test = test.reshape(-1,1)
for i in range(1000, 2000):
    data = plotdata(df, i)
    combined_array = np.append(empty, data)
    if(i % 200 == 0 or i == 4014):
        print(str(i) + " out of 4014 complete")