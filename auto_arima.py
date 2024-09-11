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

file_path = f'/sample.csv'
df = pd.read_csv(file_path,usecols=lambda x: x not in ["Const1","Const2"],encoding='shift-jis',parse_dates=['Date'], index_col='Date')

# association
# acf = plot_acf(df['Temperature'], lags=60)
# pacf = plot_pacf(df['Temperature'], lags=60)

# ADF testing
dftest = adfuller(df['Temperature'], autolag = 'AIC')
print("ADF : ",dftest[0])
print("P-Value : ", dftest[1])
print("Num Of Lags : ", dftest[2])
print("Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

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
