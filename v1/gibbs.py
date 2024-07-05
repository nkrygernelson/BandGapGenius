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

n_max = 100
elements = ["Bi", "Sb", "V", "Ta"]


file_name_dir = "data/data_formation_energy.txt"
for element in elements:
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
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            for k in range(n_max + 1):
                if i + j + k == n_max:
                    formulas.append(f"{element}{i}P{j}S{k}")
                    X_test.append(attribute_calculator.all_attributes(f"Bi{i}P{j}S{k}"))

    X_test = np.array(X_test)
    reg = RotationForestRegressor(n_estimators=10)
    reg.fit(X_train, Y_train.ravel())
    Y_pred = reg.predict(X_test)
    formula_and_energy = [(formula, energy) for formula, energy in zip(formulas, Y_pred)]
    formula_and_energy.sort(key=lambda x: x[1])

    #write the data to a file
    #if the file does not exist, create it
    file_name_dir_data = "data/predictions/data_FE_" + element + ".txt"
    if not os.path.exists(file_name_dir):
        with open(file_name_dir_data, 'w') as f:
            for formula, energy in formula_and_energy:
                f.write(formula + "," + str(energy) + "\n")
    else:
        with open(file_name_dir_data, "a") as f:
            for formula, energy in formula_and_energy:
                f.write(formula + "," + str(energy) + "\n")

        

            
    
#TaPS6


