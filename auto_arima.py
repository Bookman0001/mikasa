import io
import boto3
import pandas as pd
import numpy as np
import time
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from matplotlib import pylab as plt
from sklearn.model_selection import train_test_split
from pmdarima import model_selection
import pmdarima as pm
from sklearn.metrics import r2_score

bucket_name = 'temperature-survey'
s3_file_path = 'tokyo_100years_monthly_highest_temperature.csv'
local_file_path = '/experiment/tokyo_100years_by_month.csv'

s3 = boto3.client('s3')
response = s3.get_object(Bucket=bucket_name, Key=s3_file_path)
csv_content = response['Body'].read().decode('shift-jis')

df = pd.read_csv(io.StringIO(csv_content),usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')

# splitting data into train and test
df_train,df_test = train_test_split(df['Temperature'],test_size=0.2,shuffle=False)

# creating model
start_time = time.time()
model = pm.auto_arima(df_train,seasonal=True,m=12,trace=True)
end_time = time.time()
model.summary()

# plot
first_day_of_prediction = df_test.keys()[0]
pred_original = model.predict(n_periods=df_test.shape[0])
test_original = df['Temperature'][first_day_of_prediction:]
plt.plot(test_original)
plt.plot(pred_original, "r")

print('RMSE:',np.mean(np.abs((pred_original - test_original) / test_original)))
print('R^2: ',r2_score(pred_original, test_original))
print("training time:", end_time - start_time)
