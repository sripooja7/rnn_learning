# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"git tutorial...."
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values



plt.plot(training_set)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled=sc.fit_transform(training_set)

x_train=[]
y_train=[]

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]),1)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor=Sequential()
#ctrl+i
regressor.add(LSTM(50,return_sequence=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# adding the output layer
regressor.add(Dense(units=1))

# compiling and training
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x_train,y_train,epochs=100,batch_size=32)

regressor.save('rnn.h5')

dataset_test = pd.read_csv('Google_Stock_Price_Train.csv')

real_stock_price=dataset_test.iloc[:,1:2].values
#combine the dataset
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

#open-name of column to be obtained
#concat for combining the rows in both the files 
#data preprocessing

inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values

inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

x_test=[]

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
    
#x_test=np.reshape(x_test)
x_test = np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_stock_price=regressor.predict(x_test)

#inverse transform

predicted_stock_price=sc.inverse_transform(predicted_stock_price)

# visualisation

plt.plot(real_stock_price,color='red', label='Real Google Price')


plt.plot(predicted_stock_price,color='blue', label='Predicted Google Price')
