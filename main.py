import pickle
from pyexpat import features
import numpy as np

import streamlit as st

def get_user_inputs():
    st.sidebar.title("User Parameters")

    dict1 = {'First Owner':1, 'Second Owner':2,'Third Owner':3,'Fourth and Above Owner':4,'Test Drive Car':0}
    dict2 = {'Automatic':1, 'Manual':2}
    dict3 = {'Diesel':1, 'Petrol':2, 'CNG':3, 'LPG':4}

    year=st.sidebar.slider('Year',1985,2021,2014)
    dist=st.sidebar.slider ("km_driven",1000,2400000,50000)
    fuel=st.sidebar.selectbox("Fuel Type",('Petrol','Diesel','CNG','LPG'))
    transmission=st.sidebar.selectbox("Transmission",('Automatic','Manual'))
    owner=st.sidebar.selectbox("Owner",('First Owner','Second Owner', 'Third Owner','Fourth and Above Owner', 'Test Drive Car'))
    mileage=st.sidebar.slider ("Mileage",5,35,15)
    engine=st.sidebar.slider ("Engine",625,3500,1298)
    seat=st.sidebar.slider ("Seats",4,9,4)

    fuel = dict3[fuel]
    transmission = dict2[transmission]
    owner = dict1[owner]
    features = np.array([year,dist,fuel,transmission,owner,mileage,engine,seat]).reshape(1,-1)
    return features


if __name__=="__main__":
    st.header("Second Hand Vehicle Price Prediction")
    features = get_user_inputs()
    
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(features)
    st.write(features)
    st.write(result)



    




