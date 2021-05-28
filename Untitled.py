

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

def plotdata(df0, a):
    figure = mpl.pyplot.figure(  )
    plot   = figure.add_subplot ( 111 )
    plot.plot(df0.drop(['Holiday','Station Name'],axis=1).iloc[a,:])
    
    # draw the renderer
    figure.canvas.draw (plotdata(df,a))
 
    # Get the RGBA buffer from the figure
    w,h = figure.canvas.get_width_height()
    buf = np.fromstring ( figure.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    
    return buf
img2 = plotdata(df, 0)
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
