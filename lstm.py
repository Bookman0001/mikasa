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
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from matplotlib import pylab as plt

# fix the random number to generate reproducible RMSE and R^2
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    
# creating matrix
def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    for i in range(look_back,len(dataset)):
        data_x.append(dataset[i-look_back:i,0])
        data_y.append(dataset[i,0])    
    return np.array(data_x), np.array(data_y)


bucket_name = 'temperature-survey'
s3_file_path = 'tokyo_100years_monthly_highest_temperature.csv'
local_file_path = '/experiment/tokyo_100years_by_month.csv'
look_back = 3


s3 = boto3.client('s3')
response = s3.get_object(Bucket=bucket_name, Key=s3_file_path)
csv_content = response['Body'].read().decode('shift-jis')
df = pd.read_csv(io.StringIO(csv_content),usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')
number_of_removal_dataset = 30 * 0
temperatures = df.values[number_of_removal_dataset:]

# splitting data into train and test
train,test = train_test_split(temperatures,test_size=0.2,shuffle=False)

# scaling between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)

# creating training dataset
train_x, train_y = create_dataset(scaled_train, look_back)
train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))

# creating model with early stopping
set_seed(0)
model = Sequential()
model.add(LSTM(16, input_shape=(train_x.shape[1],1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss',verbose=1,restore_best_weights=True,min_delta=0.0001)
start_time = time.time()
model.fit(train_x, train_y, epochs=40, batch_size=1,callbacks=[early_stopping],verbose=2)
end_time = time.time()

# preparation of test
transformed_test = scaler.transform(test)
test_x,_ = create_dataset(transformed_test, look_back)
test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))

# prediction
test_predict = model.predict(test_x)
predicted_temperature = scaler.inverse_transform(test_predict)

# evaluation
actual_temperature = test[look_back:]
plt.plot(predicted_temperature,"r")
plt.plot(actual_temperature)
print('RMSE:', root_mean_squared_error(actual_temperature,predicted_temperature))
print('MAE:', mean_absolute_error(actual_temperature,predicted_temperature))
print('MAPE:', mean_absolute_percentage_error(actual_temperature,predicted_temperature))
print('R^2: ', r2_score(actual_temperature,predicted_temperature))
print("training time:", end_time - start_time)
