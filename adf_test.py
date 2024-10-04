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

# ADF testing
dftest = adfuller(df['Temperature'], autolag = 'AIC')
print("ADF : ",dftest[0])
print("P-Value : ", dftest[1])
print("Num Of Lags : ", dftest[2])
print("Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)
