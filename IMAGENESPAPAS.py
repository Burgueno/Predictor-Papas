from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from matplotlib import image

dataset =  DataFrame(pd.read_csv('potatoeseries03.csv', header=None))
print(dataset.head)

datasetnorm = preprocessing.minmax_scale(dataset, feature_range=(0, 1))
datasetnorm = DataFrame(datasetnorm)
print(datasetnorm)

import Tools as tls
from sklearn.model_selection import train_test_split
import numpy as np


IMAGE_SIZE=30
N = 200
cont = 0
cont2 = 0
lista = []
X_train = []
y_train = []
X_test = []
y_test = []
#X_train, y_train, X_test, y_test = tls.SerieToImage(datasetnorm, IMAGE_SIZE)
#print(datasetnorm)
data_train, data_test = train_test_split(datasetnorm[0], test_size=0.33, shuffle=False)
#print(len(data_train),len(data_test))
for d in data_train:
    lista.append(d)
    cont+=1
    cont2+=1
    if cont == N:
        #imageArray = preprocessing.minmax_scale(lista, feature_range=(0, 255))
        dataImage = np.array(lista).reshape(N, 1)
        cont-=1
        lista.pop(0)
        if cont2+1 == len(data_train):
            break
        X_train.append(dataImage)
        y_train.append([data_train[cont2+1]])
cont2 = 0
for d in data_test:
    lista.append(d)
    cont+=1
    cont2+=1
    if cont == N:
        #imageArray = preprocessing.minmax_scale(lista, feature_range=(0, 255))
        dataImage = np.array(lista).reshape(N, 1)
        cont-=1
        lista.pop(0)
        if cont2+1 == len(data_test):
            break
        X_test.append(dataImage)
        y_test.append([data_train[cont2+1]])
print(y_test)


tls.imageDataToJPG(X_train, y_train, 'images/train')
tls.imageDataToJPG(X_test, y_test, 'images/test')