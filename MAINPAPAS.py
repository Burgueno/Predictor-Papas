from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from PIL import Image
from keras.preprocessing import image
import CNNPAPAS as cnn
import numpy as np

listdf = []
rute = 'images'
dftrain = pd.read_csv(rute+'/train/relacion.csv',header=None)
dftest = pd.read_csv(rute+'/test/relacion.csv',header=None)
x_train=[]
x_test=[]
y_train=[]
y_test=[]
for index,row in dftrain.iterrows():
	imagen = image.load_img(rute+'/train/'+row[0],color_mode='grayscale')
	imagen = image.img_to_array(imagen)
	x_train.append(imagen)
	y_train.append([row[1]])
for index,row in dftest.iterrows():
	imagen = image.load_img(rute+'/test/'+row[0],color_mode='grayscale')
	imagen = image.img_to_array(imagen)
	x_test.append(imagen)
	y_test.append([row[1]])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))

print (x_train.shape, x_test.shape, y_train.shape, y_test.shape)

cnn.cnn_train(x_train,y_train,x_test,y_test,(1,200,1))