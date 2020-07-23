from keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from PIL import Image
from keras.preprocessing import image

def cnn_train(x_train,y_train,x_test,y_test,in_shape,neurons1=32,neurons2=64,neurons3=128,neurons4=1,kernel=(1,3),epochs=10,batch=24,dropout=0.05):
	model = Sequential((
	            Convolution2D(neurons1,kernel_size=kernel, activation='relu',input_shape=in_shape),
	            #MaxPooling2D(pool_size=(3, 1)),
	            Convolution2D(neurons2,kernel_size=kernel, activation='relu'),
	            #MaxPooling2D(pool_size=(3, 1)),
	            Flatten(),
	            Dense(neurons3, activation='relu'),
	            Dropout(dropout),
	            Dense(neurons4,activation='linear'),
	            ))     
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])

	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_test, y_test), verbose = 1)
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	#Se realizan predicciones con datos de prueba
	predictions = model.predict(x_test, 10, verbose=1)
	#Se grafican predicciones
	pyplot.figure(num=None, figsize=(18, 6), dpi=320, facecolor='w', edgecolor='k')
	pyplot.plot(y_test[1:1100,], label='Real')
	pyplot.plot(predictions[1:1100,], label='Predicción')
	pyplot.title('Valor predicción contra real')
	pyplot.xlabel('Epochs')
	pyplot.ylabel('Value')
	pyplot.legend()
	pyplot.savefig('ex')
	pyplot.show()