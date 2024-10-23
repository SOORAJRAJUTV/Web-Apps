# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:15:11 2024

@author: soora
"""

import numpy as np
import pandas as pd
import pickle

loaded_model = pickle.load(open("C:/Users/soora/OneDrive/Desktop/MLD/Heart Disease/heart_disease_model.sav",'rb'))


input_data = (37,1,4,140,207,0,0,130,1,1.5,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

if(prediction[0]==0):
  print('The person does not have a Heart Disease')
else:
  print('The person has Heart Disease')