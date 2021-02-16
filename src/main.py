import pandas as pd
from car_prediction import car_predictor, car_details
from modelling import gradient_boost, splitting_three_variables_data
import streamlit as st

def main():
    df = pd.read_csv("df_clean_model.csv")
    user_info = car_details()
    data = splitting_three_variables_data(df)
    model = gradient_boost(data[0], data[1], data[2], data[3])
    print("${}".format(car_predictor(model[2], user_info[0], user_info[1], user_info[2])))

   