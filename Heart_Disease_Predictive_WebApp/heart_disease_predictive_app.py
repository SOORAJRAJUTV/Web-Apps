# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:32 2024

@author: soora
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/soora/OneDrive/Desktop/MLD/Heart Disease/heart_disease_model.sav",'rb'))



def heart_disease_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if(prediction[0]==0):
      return 'The person does not have a Heart Disease'
    else:
      return 'The person has Heart Disease'
  
    
def main():
    
    st.title('Heart Disease Prediction System')
    
  
    
    Age = st.text_input('Age')
    Sex = st.text_input('Sex')
    Chestpaintype = st.text_input('chest pain type')
    Restingbps = st.text_input('resting bp s')
    Cholesterol= st.text_input('cholesterol')
    Fastingbloodsugar= st.text_input('fasting blood sugar')
    Restingecg= st.text_input('resting ecg')
    Maxheartrate= st.text_input('max heart rate')
    Exerciseangina= st.text_input('exercise angina')
    Oldpeak= st.text_input('oldpeak')
    STslope = st.text_input('ST slope')
    
    diagnosis = ''
    
    if(st.button('Test Result')):
        diagnosis = heart_disease_prediction((Age,Sex,Chestpaintype,Restingbps,Cholesterol,Fastingbloodsugar,Restingecg,Maxheartrate,Exerciseangina,Oldpeak,STslope))
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    