# -*- coding: utf-8 -*-
"""딥러닝_03_RNN.ipynb의 사본

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mfUlB5YZ1oBhD2zQY2qRAGnCqunfcFR8
"""

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

print(train_input.shape, test_input.shape)

len(train_input[0])

len(train_input[1])

print(train_input[0])

print(train_target)

from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

print(train_input.shape, val_input.shape)

import numpy as np
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))

import matplotlib.pyplot as plt
plt.hist(lengths)
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)

print(train_seq[0])

print(train_input[0][-10:])

print(train_seq[5])

from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape)

print(train_oh[0][0][:12])

model.summary()

val_seq = pad_sequences(val_input, maxlen=100)
print(val_seq.shape)

val_oh = keras.utils.to_categorical(val_seq)
print(val_oh.shape)

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

cp = keras.callbacks.ModelCheckpoint('my_rnn.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience =3, restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[cp, es])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

cp = keras.callbacks.ModelCheckpoint('my_rnn.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience =3, restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[cp, es])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

"""#LSTM"""

from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
print(train_seq.shape, val_seq.shape)

from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model.add(keras.layers.LSTM(8, dropout=0.3))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

cp = keras.callbacks.ModelCheckpoint('my_rnn.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience =3, restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target), callbacks=[cp, es])

"""# 주가 예측해 보기"""

!pip install pykrx

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pykrx import stock
from datetime import datetime

train_data = stock.get_market_ohlcv_by_date('20170101','20211231','005930') #삼성전자 주가
train_data.head()
print(train_data['시가'].count)

train_data.to_csv('./samsong_stock_price_train.csv')

test_data = stock.get_market_ohlcv_by_date('20220101','20220131','005930') #삼성전자 주가
test_data.to_csv('./samsong_stock_price_test.csv')

test_data.head()

train_x = train_data.iloc[:,0:1].values
print(train_x, train_x.size)

"""
Standardisation
$$X_{stand} = \frac{x - mean(x)}{standard deviation (x)}$$

Normalisation
$$X_{norm} = \frac{x - min(x)}{max(x) - min(x)} $$
  """

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(train_x)
print(training_set_scaled)

X_train = []
y_train = []
for i in range(60, train_x.size):
  X_train.append(training_set_scaled[i-60:i, 0])
  y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

