# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:42:31 2024

@author: soora
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st


loaded_model = pickle.load(open("C:/Users/soora/OneDrive/Desktop/MLD/Fraud Detection/fraud_trained_model.sav",'rb'))
scaler = pickle.load(open("C:/Users/soora/OneDrive/Desktop/MLD/Fraud Detection/scaler.sav",'rb'))


def fraud_detection(input_data):
    
   input_data_as_numpy_array = np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   input_data_scaled = scaler.transform(input_data_reshaped)
   prediction = loaded_model.predict(input_data_scaled)

   if(prediction[0]==0):
      return 'The Transaction is Legit'
   else:
      return 'The Transaction is Fraudulent'
  
    
def main():
    
    st.title('Fraud Detection System')
    
    CUSTOMER_ID = st.text_input('CUSTOMER ID')
    TERMINAL_ID = st.text_input('TERMINAL ID')
    TX_AMOUNT = st.text_input('TX AMOUNT')
    TX_TIME_SECONDS = st.text_input('TX TIME_SECONDS')
    TX_TIME_DAYS= st.text_input('TX TIME_DAYS')
    TX_FRAUD_SCENARIO= st.text_input('TX FRAUD_SCENARIO')
    
   
    diagnosis = ''
    
    if(st.button('Test Result')):
        diagnosis = fraud_detection((CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_TIME_SECONDS,TX_TIME_DAYS,TX_FRAUD_SCENARIO))
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()