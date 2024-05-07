# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:32:19 2024

@author: Amaan
"""
import numpy as np
import pickle
import streamlit as st 

loaded_model = pickle.load(open('trained_model.pkl','rb'))

def house_price_prediction(input_data):
  #changing input data to numpy array
  input_data_as_numpy_array=np.asarray(input_data, dtype=np.float64)
  #reshaping the array
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  prediction = loaded_model.predict(input_data_reshaped)
  return(prediction[0])


def main():
  st.title ('House Price Prediction Web App')

  total_sqft=st.text_input('Total sqft')
  bath=st.text_input('Total Bathrooms')
  size_bhk=st.text_input('Size in BHK')
  price_per_sqft=st.text_input('Price per sqft')

  price= ""
  #code for prediction

  if st.button("Predict Price"):
    price=house_price_prediction([total_sqft,bath,size_bhk,price_per_sqft])
  
  st.success(price)


if __name__=='__main__':
  main()



