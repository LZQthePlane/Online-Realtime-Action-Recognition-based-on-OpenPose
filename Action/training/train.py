import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

# load data
raw_data = pd.read_csv('data.csv', header=0)
dataset = raw_data.values
X = dataset[:, 0:36].astype(float)
Y = dataset[:, 36]

# 将类别编码为数字
encoder = LabelEncoder()
encoder_Y = encoder.fit_transform(Y)
print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

# build keras model
# model = Sequential()
# model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=16, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=4, activation='softmax'))

# training
# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test))
# model.save('framewise_recognition.h5')

# test
model = load_model('framewise_recognition.h5')

# test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
#               0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
#               0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
#               0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
# test_np = np.array(test_input)
# test_np = test_np.reshape(-1, 36)

test_np = np.array(X[1033]).reshape(-1, 36)
if test_np.size > 0:
    pred = np.argmax(model.predict(test_np))
    init_label = encoder.inverse_transform(pred)
    print(init_label)