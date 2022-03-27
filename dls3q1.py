from keras import backend as K
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import time
from time import time
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
 

mbs_size = 128

model = Sequential()
model.add(Dense(1000, input_shape = (32,32,3), activation='relu', kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01)))
#model.add(BatchNormalization())
model.add(Dense(1000, activation='relu', kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01)))
#model.add(BatchNormalization()))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics="mse")
start = time()
adam_his = model.fit(x_train, y_train, epochs=200, batch_size=mbs_size, )
end = time() - start
print("\n Time taken to fit is",end)

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics="mse")


