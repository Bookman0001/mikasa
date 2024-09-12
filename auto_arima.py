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


bucket_name = 'sample'
s3_file_path = 'sample.csv'
local_file_path = 'sample.csv'

s3 = boto3.client('s3')
response = s3.get_object(Bucket=bucket_name, Key=s3_file_path)
csv_content = response['Body'].read().decode('shift-jis')

df = pd.read_csv(io.StringIO(csv_content),usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')

# decreasing data amount
df = df.iloc[800:]

# association
# acf = plot_acf(df['Temperature'], lags=60)
# pacf = plot_pacf(df['Temperature'], lags=60)

# ADF testing
# df.plot(figsize=(18,6))
# dftest = adfuller(df['Temperature'], autolag = 'AIC')
# print("原系列1. ADF : ",dftest[0])
# print("原系列2. P-Value : ", dftest[1])
# print("原系列3. Num Of Lags : ", dftest[2])
# print("原系列4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
# print("原系列5. Critical Values :")
# for key, val in dftest[4].items():
    # print("\t",key, ": ", val)
    
# splitting data into train and test
df_train,df_test = train_test_split(df['Temperature'],test_size=0.2,shuffle=False)

# creating model
start_time = time.time()
model = pm.auto_arima(df_train,seasonal=True,m=12,trace=True)
end_time = time.time()
model.summary()
# r_diff.plot_diagnostics(lags=20);

# plot
first_day_of_prediction = df_test.keys()[0]
pred_original = model.predict(n_periods=df_test.shape[0])
test_original = df['Temperature'][first_day_of_prediction:]
plt.plot(test_original)
plt.plot(pred_original, "r")

print(np.mean(np.abs((pred_original - test_original) / test_original)))
print(r2_score(pred_original, test_original))
print("training time:", end_time - start_time)