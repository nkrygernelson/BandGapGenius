import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
import v1.attribute_calculator as attribute_calculator
from aeon.regression.sklearn import RotationForestRegressor
import os


scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

def generate_data(file_name_dir):
    with open(file_name_dir, "r") as f:
        data = f.readlines()
        for line in data:
            formula, property = line.split(",")
            property = float(property)
            attribute = attribute_calculator.all_attributes(formula)
            #if the file does not exist, create it
            if not os.path.exists(file_name_dir):
                with open(file_name_dir[:-4]+"_attributes.txt", 'w') as f:
                    f.write(str(attribute) + ";" + str(property) + "\n")
            else:
                with open(file_name_dir[:-4]+"_attributes.txt", "a") as f:
                    f.write(str(attribute) + ";" + str(property) + "\n")



def split_data_scale(file_name_dir):
    X = []
    Y = []
    with open(file_name_dir, "r") as f:
        data = f.readlines()
        for line in data:
            attributes, band_gap = line.split(";")
            attributes = np.array([float(i) for i in attributes[1:-1].split(",")])
            band_gap = float(band_gap)
            X.append(attributes)
            Y.append(band_gap)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    
    # Scale the data
    X = scaler_X.fit_transform(X)
    Y = scaler_Y.fit_transform(Y)

    # Split the data
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.4, random_state=42)
    return X_train, Y_train, X_test, Y_test

def split_data(file_name_dir):
    X = []
    Y = []
    with open(file_name_dir, "r") as f:
        data = f.readlines()
        for line in data:
            attributes, band_gap = line.split(";")
            attributes = np.array([float(i) for i in attributes[1:-1].split(",")])
            band_gap = float(band_gap)
            X.append(attributes)
            Y.append(band_gap)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    

    # Split the data
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.4, random_state=42)
    return X_train, Y_train, X_test, Y_test

def NN(X_train, Y_train, X_test,Y_test):
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
        Y_train,
        epochs=100,
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.2
    )

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss: Training vs Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predict the band gaps of the test data
    Y_pred = scaler_Y.inverse_transform(model.predict(X_test))
    Y_test = scaler_Y.inverse_transform(Y_test)
    print("length y_pred")
    print(len(Y_pred))
    Y_pred_single = model.predict(X_test[0].reshape(1, -1))


    # Plot the predicted band gaps against the actual band gaps
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(-2.3, 1, 100), np.linspace(-2.3, 1, 100), color='red')
    plt.xlabel('Actual Formation Energy')
    plt.ylabel('Predicted Formation Energy')
    plt.title('Actual Formation Energy vs Predicted Formation Energy')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat 

def random_forest(X_train,Y_train,X_test,Y_test):
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, Y_train.ravel())
    Y_pred = rfr.predict(X_test)
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(-2.3, 1, 100), np.linspace(-2.3, 1, 100), color='red')
    plt.xlabel('Actual Formation Energy')
    plt.ylabel('Predicted Formation Energy')
    plt.title('Actual Formation Energy vs Predicted Formation Energy')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat



def grid_search_random_forest(X_train,Y_train,X_test,Y_test):
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = sklearn.model_selection.GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    Y_pred = grid_search.predict(X_test)
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(-2.3, 1, 100), np.linspace(-2.3, 1, 100), color='red')
    plt.xlabel('Actual Formation Energy')
    plt.ylabel('Predicted Formation Energy')
    plt.title('Actual Formation Energy vs Predicted Formation Energy')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat


def rotating_forest(X_train,Y_train, X_test,Y_test):
    reg = RotationForestRegressor(n_estimators=10)
    reg.fit(X_train, Y_train.ravel())
    Y_pred = reg.predict(X_test)
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(-2.3, 1, 100), np.linspace(-2.3, 1, 100), color='red')
    plt.xlabel('Actual Formation Energy')
    plt.ylabel('Predicted Formation Energy')
    plt.title('Actual Formation Energy vs Predicted Formation Energy')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat

#print(NN(X_test,Y_test))


elements_list = [
    ["P", "S"],
    ["As", "S"],
    ["As", "Se"],
    ["P", "Se"],
    ["P", "Te"],
    ["Sb", "Te"]
]
RoF_metrics = {}
RF_metrics = {}
for include_list in elements_list:
    file_name_dir = "data/data_FE_" + "_".join(include_list) + ".txt"
    generate_data(file_name_dir)
    x_train, y_train, X_test, Y_test = split_data(file_name_dir[:-4]+"_attributes.txt")
    print("_".join(include_list))
    #number of training samples
    train_number = len(x_train)
    #number of test samples
    test_number = len(X_test)
    RoF_metrics["_".join(include_list)+"_r2"], RoF_metrics["_".join(include_list)+"_mse"] = rotating_forest(x_train, y_train, X_test,Y_test)
    #add the number of training and test samples to the dictionary
    RoF_metrics["_".join(include_list)+"_train_number"] = train_number
    RoF_metrics["_".join(include_list)+"_test_number"] = test_number
    RF_metrics["_".join(include_list)+"_r2"], RF_metrics["_".join(include_list)+"_mse"] = random_forest(x_train, y_train, X_test,Y_test)
    RF_metrics["_".join(include_list)+"_train_number"] = train_number
    RF_metrics["_".join(include_list)+"_test_number"] = test_number
print("Rotation Forest")
print(RoF_metrics)
print("Random Forest")
print(RF_metrics)
