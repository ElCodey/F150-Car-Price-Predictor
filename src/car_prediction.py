
import numpy as np


def car_details():
    condition_options = ["fair", "good", "excellent", "like_new", "new"]
    car_condition = input("What condition is the car in? fair/good/excellent/like_new/new: ").lower()
    
    if car_condition not in condition_options:
        print("Please input a valid condition from the given options. ")
        return car_details()
    
    car_year = float(input("What is the year of the car? (Between 2010-2020): "))
    car_odo = float(input("How many miles has it done? "))
    
    
                          
    return car_year, car_odo, car_condition


def car_predictor(model, year, odo, condition):
    if condition == "fair":
        data = np.array([year, odo, 0, 1, 0, 0, 0]).reshape((1, -1))
        prediction = model.predict(data)
        return prediction[0].round(2)
    elif condition == "excellent":
        data = np.array([year, odo, 1, 0, 0, 0, 0]).reshape((1, -1))
        prediction = model.predict(data)
        return prediction[0].round(2)
    elif condition == "good":
        data = np.array([year, odo, 0, 0, 1, 0, 0]).reshape((1, -1))
        prediction = model.predict(data)
        return prediction[0].round(2)
    elif condition == "like_new":
        data = np.array([year, odo, 0, 0, 0, 1, 0]).reshape((1, -1))
        prediction = model.predict(data)
        return prediction[0].round(2)
    elif condition == "new":
        data = np.array([year, odo, 0, 0, 0, 0, 1]).reshape((1, -1))
        prediction = model.predict(data)
        return prediction[0].round(2)
