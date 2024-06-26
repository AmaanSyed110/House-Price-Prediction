# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:41:06 2024

@author: Amaan
"""

import numpy as np
import pickle

#loading the saved model
loaded_model=pickle.load(open('D:/Git-Hub projects/House-Price-Prediction/trained_model.pkl','rb'))

input_data = (1095,3,3,3569)
#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data, dtype=np.float64)

#reshaping the array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)