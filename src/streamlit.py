import pandas as pd
from car_prediction import car_predictor
from modelling import gradient_boost, splitting_three_variables_data
import streamlit as st

#Streamlit App
#Function to load data, run the model and st command to cache the model for quicker app use

def streamlit_data_and_model():
    df = pd.read_csv("df_clean_model.csv")
    data = splitting_three_variables_data(df)
    model = gradient_boost(data[0], data[1], data[2], data[3])

    return model[2]
    
model = streamlit_data_and_model()    

st.title("Ford F150 Car Price Predictor")


car_year = st.selectbox("What is the year of your car?", ("2010", "2011", "2012", "2013", "2014", "2015",
                                                          "2016", "2017", "2018", "2019", "2020"))
car_miles = st.text_input("How many miles has your car done?")
car_condition = st.selectbox("What is the car condition?", ("Fair", "Good", "Excellent", "Like_New", "New"))

try:
    year = float(car_year)
    miles = float(car_miles)
    condition = car_condition.lower()
    prediction = car_predictor(model, year, miles, condition)
    st.write("${}".format(prediction))
except:
    pass