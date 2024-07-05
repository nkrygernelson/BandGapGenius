import matplotlib.pyplot as plt
import ternary
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
import v1.attribute_calculator as attribute_calculator
import os
from aeon.regression.sklearn import RotationForestRegressor




file_name_dir = "data/big_data_band_gap.txt"
X_train = []
Y_train = []
with open(file_name_dir[:-4]+"_attributes.txt", "r") as f:
    data = f.readlines()
    for line in data:
        attributes, Fe = line.split(";")
        attributes = np.array([float(i) for i in attributes[1:-1].split(",")])
        Fe = float(Fe)
        X_train.append(attributes)
        Y_train.append(Fe)
X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape(-1, 1)
#make the test data
X_test = []
formulas = []

X_test.append(attribute_calculator.all_attributes("SrTiO3"))

X_test = np.array(X_test)
reg = RotationForestRegressor(n_estimators=10)
reg.fit(X_train, Y_train.ravel())
Y_pred = reg.predict(X_test)
print(Y_pred)

    

            
    
#TaPS6


