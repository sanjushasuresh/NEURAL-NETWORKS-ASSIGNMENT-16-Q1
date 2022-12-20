# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 10:23:26 2022

@author: LENOVO
"""

# Method 1

# pip install keras 
# pip install tensorflow

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv("gas_turbines.csv")
df.head()
df.info()


# Splitting
X1=df.iloc[:,:7]
X2=df.iloc[:,8:]
X=pd.concat([X1,X2],axis=1)
X.columns
X.dtypes

Y=df["TEY"]


model=Sequential()
model.add(Dense(15,input_dim=10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["mean_squared_error"])


history=model.fit(X,Y,validation_split=0.33,epochs=250,batch_size=10)
scores=model.evaluate(X,Y)
print("%s: %.2f%%"%(model.metrics_names[1],scores[1]*100))
history.history.keys()
# mean_squared_error = 17989.7324
 

import matplotlib.pyplot as plt
plt.plot(history.history["mean_squared_error"])
plt.plot(history.history["val_loss"])
plt.title("model mean quared error")
plt.ylabel("mean_squared_error")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()


#=============================================================================================#


# Method 2

import numpy as np
import pandas as pd

df=pd.read_csv("gas_turbines.csv")
df.head()
df.isnull().sum()
df.duplicated()
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.columns
df.dtypes


# Splitting
X1=df.iloc[:,:7]
X2=df.iloc[:,8:]
X=pd.concat([X1,X2],axis=1)
X.dtypes
X.columns

Y=df["TEY"]


# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(30,30))
mlp.fit(X_train,Y_train)
pred_train=mlp.predict(X_train)
pred_test=mlp.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
rs1=r2_score(Y_train,pred_train)
mse1=mean_squared_error(Y_train,pred_train)
# rs1=0.99611 (99%)
rs2=r2_score(Y_test,pred_test)
mse2=mean_squared_error(Y_train,pred_train)
# rs2=0.99601 (99%)
# mse1 & mse2 = 1.257307253310199
