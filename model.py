import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.model_selection
from tensorflow import keras
import attribute_calculator
import rotation_forest



scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

def generate_data():
    with open("data/data.txt", "r") as f:
        data = f.readlines()
        for line in data:
            formula, band_gap = line.split(",")
            band_gap = float(band_gap)
            attribute = attribute_calculator.all_attributes(formula)
            with open("data/data_attributes.txt", "a") as f:
                f.write(str(attribute) + ";" + str(band_gap) + "\n")

#generate_data()

def split_data():
    X = []
    Y = []
    with open("data/data_attributes.txt", "r") as f:
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

X_train, Y_train, X_test, Y_test = split_data()
#create a class containing this model and other models

def NN(X_test,Y_test):    

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

    print(X_test[0])
    Y_pred_single = model.predict(X_test[0].reshape(1, -1))

    # Plot the predicted band gaps against the actual band gaps
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
    plt.xlabel('Actual band gap')
    plt.ylabel('Predicted band gap')
    plt.title('Actual band gap vs Predicted band gap')
    plt.show()

NN(X_test,Y_test)

    

