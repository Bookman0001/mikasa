# condition look_back = 3, neuron = 16, dense = 1, min_delta=0.0001
import tensorflow as tf
import numpy as np
import random
import os

import io
import boto3
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense,GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pylab as plt

# fix the random number to generate reproducible RMSE and R^2
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


bucket_name = 'temperature-survey'
s3_file_path = 'tokyo_100years_monthly_highest_temperature.csv'
local_file_path = '/experiment/tokyo_100years_by_month.csv'

s3 = boto3.client('s3')
response = s3.get_object(Bucket=bucket_name, Key=s3_file_path)
csv_content = response['Body'].read().decode('shift-jis')

df = pd.read_csv(io.StringIO(csv_content),usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')

# scaling between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
df_scale = pd.DataFrame(scaler.fit_transform(df),index=df.index,columns=df.columns)

# splitting data into train and test
df_scale_train,df_scale_test = train_test_split(df_scale['Temperature'],test_size=0.2,shuffle=False)

# creating matrix
def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    print(range(len(dataset)-look_back-1))
    for i in range(len(dataset)-look_back-1):
        d = dataset[i:(i+look_back)]
        data_x.append(d)
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)

look_back = 3
train_x, train_y = create_dataset(df_scale_train, look_back)
test_x, test_y = create_dataset(df_scale_test, look_back)

# converting to [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# creating model with early stopping
set_seed(0)
model = Sequential()
model.add(GRU(16, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss',verbose=1,restore_best_weights=True,min_delta=0.001)
start_time = time.time()
model.fit(train_x, train_y, epochs=40, batch_size=1,callbacks=[early_stopping],verbose=2)
end_time = time.time()

# prediction
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# inversing
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# plot of prediction
predicted_temperature = pd.DataFrame(test_predict)
actual_temperature = pd.DataFrame(test_y).transpose()

plt.plot(predicted_temperature,"r")
plt.plot(actual_temperature)
print('RMSE:', np.mean(np.abs((predicted_temperature - actual_temperature) / actual_temperature)))
print('R^2: ', r2_score(predicted_temperature, actual_temperature))
print("training time:", end_time - start_time)
