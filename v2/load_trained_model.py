import numpy as np
import pandas as pd
from CBFV.composition import generate_features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import os
from time import time
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from aeon.regression.sklearn import RotationForestRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import joblib
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
prop  = "FE"
PATH = os.getcwd()

train_path =  PATH+"/v2/data/big_data_training/fe_train.csv"
val_path = PATH+"/v2/data/big_data_training/fe_val.csv"
test_path = PATH+"/v2/data/big_data_training/fe_test.csv"

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

print(f'df_train DataFrame shape: {df_train.shape}')
print(f'df_val DataFrame shape: {df_val.shape}')
print(f'df_test DataFrame shape: {df_test.shape}')

print('DataFrame column names before renaming:')
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)

rename_dict = {prop : 'target'}
df_train = df_train.rename(columns=rename_dict)
df_val = df_val.rename(columns=rename_dict)
df_test = df_test.rename(columns=rename_dict)


print('\nDataFrame column names after renaming:')
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)

X_train_unscaled, y_train, formulae_train, skipped_train = generate_features(df_train, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
X_val_unscaled, y_val, formulae_val, skipped_val = generate_features(df_val, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
X_test_unscaled, y_test, formulae_test, skipped_test = generate_features(df_test, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)

print(f'X_train_unscaled shape: {X_train_unscaled.shape}')
print(f'y_train shape: {y_train.shape}')

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_unscaled)
X_val = scaler.transform(X_val_unscaled)
X_test = scaler.transform(X_test_unscaled)

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


# Instantiate the model again using the parameters
#load a model from the models folder
def load_model(model_name):
    model = joblib.load(f'v2/models/{model_name}.pkl')
    return model



def evaluate_model(model, X, y_act):
    y_pred = model.predict(X)
    r2 = r2_score(y_act, y_pred)
    mae = mean_absolute_error(y_act, y_pred)
    rmse_val = mean_squared_error(y_act, y_pred, squared=False)
    return r2, mae, rmse_val
def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])
    xy_min = np.min([np.min(act), np.min(pred)])

    plot = plt.figure(figsize=(6,6))
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', label='ideal')
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(act))
        plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')
    plt.axis('scaled')
    plt.xlabel(f'Actual {label}')
    plt.ylabel(f'Predicted {label}')
    plt.title(f'{type(model).__name__}, r2: {r2_score(act, pred):0.4f}')
    plt.legend(loc='upper left')
    plt.savefig(f'v2/figures/BEST_{type(model).__name__}_pred_act.png')
    
    return plot

model = load_model('svr')
y_act_test = y_test
y_pred_test = model.predict(X_test)

r2, mae, rmse = evaluate_model(model, X_test, y_test)
print(f'r2: {r2:0.4f}')
print(f'mae: {mae:0.4f}')
print(f'rmse: {rmse:0.4f}')

plot = plot_pred_act(y_act_test, y_pred_test, model, reg_line=True, label='FE')

def check_for_simplifcation(previous_data, ijk):
    for data in previous_data:

        if data[0] != 0:
            k = ijk[0]/data[0]
        elif data[1] != 0:
            k = ijk[1]/data[1]
        elif data[2] != 0:
            k = ijk[2]/data[2]
        
        if all([data[0]*k == ijk[0], data[1]*k == ijk[1], data[2]*k == ijk[2]]):
            return True
    return False

    
n_max = 15
elements = ["Ag","Al","Au","B","Ba","Bi","Ca","Cd","Co","Cr","Cs","Cu","Fe","Ga","Ge","K","Hf","Hg","In","Ir","La","Li","Mg","Mn","Mo","Na","Nb","Ni","Os","Pb","Pd","Pt","Rb","Re","Rh","Ru","Sb","Sc","Sr","Sn","Ta","Tc","Ti","Tl","W","Y","Zr","Zn"]

    
for element in elements:
    coeff_list = []
    formulas = []
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            for k in range(n_max + 1):
                #skip the case where all the values are 0
                if i == 0 and j == 0 and k == 0:
                    continue
                else:
                    if i + j + k <= 20:
                        if not check_for_simplifcation(coeff_list, [i, j, k]):
                            formula = f"{element}{i}P{j}S{k}"
                            formulas.append(formula)
                            coeff_list.append([i, j, k])



    target = [0 for i in range(len(formulas))]
    #make a dataframe from the formulas and target
    df = pd.DataFrame({"formula": formulas, "target": target})
    print(df['formula'])
    X_test_unscaled, y_test, formulae_test, skipped_test = generate_features(df, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
    X_test = scaler.transform(X_test_unscaled)
    X_test = normalize(X_test)              
    X_test = np.array(X_test)

    y_pred = model.predict(X_test)
    y_pred = y_pred.tolist()
    ternary_data = pd.DataFrame({"formula": formulas, "target": y_pred})
    ternary_data.to_csv(f"v2/data/ternary/big_MP/ternary_data_{element}.csv", index=False)



    