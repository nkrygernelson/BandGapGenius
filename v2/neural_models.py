import numpy as np
import pandas as pd
from CBFV.composition import generate_features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import os
from time import time

import matplotlib.pyplot as plt

import joblib
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from tensorflow import keras
from sklearn import metrics
prop  = "band_gap"
PATH = os.getcwd()

train_path =  PATH+"/v2/data/big_data_training/bg_train.csv"
val_path = PATH+"/v2/data/big_data_training/bg_val.csv"
test_path = PATH+"/v2/data/big_data_training/bg_test.csv"

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



#save the scaler
joblib.dump(scaler, PATH+'/v2/models/fe_scaler.pkl')





print("Length of X_train and X_test")
print(len(X_train))
print(len(X_test))    

# Create a simple neural network model using keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_dim=X_train.shape[1], activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

batch_size = 16

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)
#save the model
model.save('v2/models/fe_big_data_nn_model.keras')
# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss: Training vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(PATH+f'/v2/figures/Neural_Network__bg_train_val_loss.png')

# Predict the band gaps of the test data
#load the model 
model = keras.models.load_model('v2/models/fe_big_data_nn_model.keras')

label = "band_gap"
Y_pred = model.predict(X_test).reshape(-1)
xy_max = np.max([np.max(y_test), np.max(Y_pred)])
xy_min = np.min([np.min(y_test), np.min(Y_pred)])

plot = plt.figure(figsize=(6,6))
plt.plot(y_test, Y_pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', label='ideal')

polyfit = np.polyfit(y_test, Y_pred, deg=1)
reg_ys = np.poly1d(polyfit)(np.unique(y_test))
plt.plot(np.unique(y_test), reg_ys, alpha=0.8, label='linear fit')
plt.axis('scaled')
plt.xlabel(f'Actual {label}')
plt.ylabel(f'Predicted {label}')
plt.title(f'Neural Network, r2: {metrics.r2_score(y_test, Y_pred):0.4f}')
plt.legend(loc='upper left')
plt.savefig(PATH+f'/v2/figures/Neural_Network_pred_act.png')

print("Metrics")
print(f"r2: {metrics.r2_score(y_test, Y_pred)}")
print(f"MAE: {metrics.mean_absolute_error(y_test, Y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, Y_pred)}")



''' 
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
    ternary_data.to_csv(f"v2/data/ternary/big_data_ternary/NN_ternary_data_{element}.csv", index=False)
'''  