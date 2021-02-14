import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor



def linear_regression(x_train, x_test, y_train, y_test, v=2):
    if v == 2:
        print("Linear Regression model for 2 variables.")
    else:
        print("Linear Regression model for 3 variables.")

    linear_model = LinearRegression()
    #Fit model
    model = linear_model.fit(x_train, y_train)
    #Predict Data
    y_pred = model.predict(x_test)
    #Calculating overall loss of the model
    linear_loss = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Overall loss is: ${}".format(linear_loss))
    #Accuracy in %
    accuracy = (sum(y_test) - linear_loss) / sum(y_test) * 100
    print("Overall accuracy in percentage is {}%".format(accuracy.round(3)))

    return linear_loss, accuracy


def gradient_boost(x_train, x_test, y_train, y_test, v=2):
    if v == 2:
        print("Gradient Boost Regression model for 2 variables.")
    else:
        print("Gradient Boost Regression model for 3 variables.")

    gbr_model = GradientBoostingRegressor()
    #Fitting model
    gbr_model = gbr_model.fit(x_train, y_train)
    #Predicitng gradient boosting
    y_gbr_predict = gbr_model.predict(x_test)
    #Calculating loss for GBR
    gbr_loss = np.sqrt(mean_squared_error(y_test, y_gbr_predict))
    print("Overall loss is: ${}".format(gbr_loss))
    #Accuracy in %
    accuracy = (sum(y_test) - gbr_loss) / sum(y_test) * 100
    print("Overall accuracy in percentage is {}%".format(accuracy.round(3)))

    #Return the model as well to use to predict user input
    return gbr_loss, accuracy, gbr_model


def splitting_two_variables_data(data):
    x = data[["year", "odometer"]]
    y= data["price"]
    #Splitting dataset into test and train
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    #Return train and test data
    return X_train, X_test, y_train, y_test

def splitting_three_variables_data(data):
    #Using condition
    X_3v = data[["year", "odometer", "excellent", "fair", "good", "like_new", "new"]]
    y_3v = data["price"]
    #Splitting the training set again with new variables
    X_train_3v, X_test_3v, y_train_3v, y_test_3v = train_test_split(X_3v, y_3v, test_size = 0.3)
    return X_train_3v, X_test_3v, y_train_3v, y_test_3v

