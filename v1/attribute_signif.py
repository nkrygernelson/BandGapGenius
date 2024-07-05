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
from sklearn.inspection import permutation_importance


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

def split_data_scale():
    X = []
    Y = []
    with open("data/data_formation_energy_attributes.txt", "r") as f:
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

def split_data():
    X = []
    Y = []
    with open("data/data_formation_energy_attributes.txt", "r") as f:
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
X_train, Y_train, X_test, Y_test = split_data_scale()
def NN(X_test,Y_test):
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
    plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
    plt.xlabel('Actual band gap')
    plt.ylabel('Predicted band gap')
    plt.title('Actual band gap vs Predicted band gap')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat 

def random_forest(X_test,Y_test):
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, Y_train.ravel())
    Y_pred = rfr.predict(X_test)
    plt.scatter(Y_test, Y_pred)
    #plot the line y = x
    plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
    plt.xlabel('Actual band gap')
    plt.ylabel('Predicted band gap')
    plt.title('Actual band gap vs Predicted band gap')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat



def grid_search_random_forest(X_test,Y_test):
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
    plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
    plt.xlabel('Actual band gap')
    plt.ylabel('Predicted band gap')
    plt.title('Actual band gap vs Predicted band gap')
    plt.show()
    metric_mat = []
    metric_mat.append(metrics.r2_score(Y_test, Y_pred))
    metric_mat.append(metrics.mean_absolute_error(Y_test,Y_pred))
    return metric_mat


def rotating_forest(X_test,Y_test):
    reg = RotationForestRegressor(n_estimators=10)
    reg.fit(X_train, Y_train.ravel())
    Y_pred = reg.predict(X_test)
   
    '''
    plt.scatter(Y_test, Y_pred)
    plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
    plt.xlabel('Actual band gap')
    plt.ylabel('Predicted band gap')
    plt.title('Actual band gap vs Predicted band gap')
    plt.show()
    '''
    
    return Y_pred

X_train, Y_train, X_test, Y_test = split_data()


# Train the model
#model = RotationForestRegressor()
#model.fit(X_train, Y_train.ravel())

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

# Plot the training and validation los

# Predict the band gaps of the test data

# Evaluate the baseline performance
baseline_predictions = model.predict(X_test)
#quick plot

plt.scatter(Y_test, baseline_predictions)
plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), color='red')
plt.show()
baseline_mse = metrics.mean_squared_error(Y_test, baseline_predictions)

scoring_function = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
# Compute permutation importance
perm_importance = permutation_importance(model, X_test, Y_test,scoring=scoring_function, n_repeats=10, random_state=42)

# Print feature importances
feature_names = [
    # Stoichiometric attributes
    "Lp_norm_p2",
    "Lp_norm_p3",
    "Lp_norm_p5",
    "Lp_norm_p7",

    # Elemental attributes
    "atomic_number_fwa",
    "atomic_number_std",
    "atomic_number_min",
    "atomic_number_max",
    "atomic_number_range",

    "row_fwa",
    "row_std",
    "row_min",
    "row_max",
    "row_range",

    "column_fwa",
    "column_std",
    "column_min",
    "column_max",
    "column_range",

    "d_valence_electrons_fwa",
    "d_valence_electrons_fwd",
    "d_valence_electrons_min",
    "d_valence_electrons_max",
    "d_valence_electrons_range",

    "d_unfilled_states_fwa",
    "d_unfilled_states_fwd",
    "d_unfilled_states_min",
    "d_unfilled_states_max",
    "d_unfilled_states_range",

    "s_unfilled_states_fwa",
    "s_unfilled_states_fwd",
    "s_unfilled_states_min",
    "s_unfilled_states_max",
    "s_unfilled_states_range",

    "p_unfilled_states_fwa",
    "p_unfilled_states_fwd",
    "p_unfilled_states_min",
    "p_unfilled_states_max",
    "p_unfilled_states_range",

    "d_unfilled_states_fwa",
    "d_unfilled_states_fwd",
    "d_unfilled_states_min",
    "d_unfilled_states_max",
    "d_unfilled_states_range",

    "f_unfilled_states_fwa",
    "f_unfilled_states_fwd",
    "f_unfilled_states_min",
    "f_unfilled_states_max",
    "f_unfilled_states_range",

    "s_valence_electrons_fwa",
    "s_valence_electrons_fwd",
    "s_valence_electrons_min",
    "s_valence_electrons_max",
    "s_valence_electrons_range",

    "p_valence_electrons_fwa",
    "p_valence_electrons_fwd",
    "p_valence_electrons_min",
    "p_valence_electrons_max",
    "p_valence_electrons_range",

    "d_valence_electrons_fwa",
    "d_valence_electrons_fwd",
    "d_valence_electrons_min",
    "d_valence_electrons_max",
    "d_valence_electrons_range",

    "f_valence_electrons_fwa",
    "f_valence_electrons_fwd",
    "f_valence_electrons_min",
    "f_valence_electrons_max",
    "f_valence_electrons_range",

    "atomic_weight_fwa",
    "atomic_weight_fwd",
    "atomic_weight_min",
    "atomic_weight_max",
    "atomic_weight_range",

    "covalent_radius_fwa",
    "covalent_radius_fwd",
    "covalent_radius_min",
    "covalent_radius_max",
    "covalent_radius_range",

    "electronegativity_fwa",
    "electronegativity_fwd",
    "electronegativity_min",
    "electronegativity_max",
    "electronegativity_range",

    "melting_point_fwa",
    "melting_point_fwd",
    "melting_point_min",
    "melting_point_max",
    "melting_point_range",

    "molar_volume_fwa",
    "molar_volume_fwd",
    "molar_volume_min",
    "molar_volume_max",
    "molar_volume_range",

    "mendeleev_number_fwa",
    "mendeleev_number_fwd",
    "mendeleev_number_min",
    "mendeleev_number_max",
    "mendeleev_number_range"]

for name, importance in zip(feature_names, perm_importance.importances_mean):
    print(f"{name}: {importance:.4f}")
sorted_idx = perm_importance.importances_mean.argsort()

# Plot
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.show()