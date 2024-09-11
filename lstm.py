import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM

file_path = f'/sample.csv'
df = pd.read_csv(file_path,usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')

scaler = MinMaxScaler(feature_range=(0, 1))
df_scale = pd.DataFrame(scaler.fit_transform(df),index=df.index,columns=df.columns)

df_scale_train,df_scale_test = train_test_split(df_scale['Temperature'],test_size=0.2,shuffle=False)

def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    print(range(len(dataset)-look_back-1))
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        data_x.append(a)
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)

look_back = 3
train_x, train_y = create_dataset(df_scale_train, look_back)
test_x, test_y = create_dataset(df_scale_test, look_back)

train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=10, batch_size=1, verbose=2)

# Prediction
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])


predicted_temperature = pd.DataFrame(test_predict)
actual_temperature = pd.DataFrame(test_y).transpose()

plt.plot(predicted_temperature,"r")
plt.plot(actual_temperature)
print('RMSE:', np.mean(np.abs((predicted_temperature - actual_temperature) / actual_temperature)))
print('R^2: ', r2_score(predicted_temperature, actual_temperature))
