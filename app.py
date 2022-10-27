import streamlit as st
import numpy as np
from prediction import pred
import joblib
from tensorflow import keras

model = keras.models.load_model('model')

def main():
    st.header("Patient Survival Detection")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTB7OCwsuNDymcNVAHOEc9pd1GV3U6sVJUquA&usqp=CAU")
    st.write('Please input the data below')

    i = st.number_input('apache_4a_hospital_death_prob',)
    j = st.number_input('apache_4a_icu_death_prob',)
    k = st.number_input('age',)
    l = st.number_input('d1_spo2_min',)
    m = st.number_input('d1_resprate_max',)
    n = st.number_input('d1_heartrate_min',)
    input = np.array([[i,j,k,l,m,n]])
    print(type(i))
    print(input)
    
    
    if st.button("predict"):
        st.write('predicting patients survival')
        predi = pred(model,[i,j,k,l,m,n])
        st.success('The predicted is ' + 'alive' if predi==0 else 'dead')

     
if __name__ == '__main__':

    main()

    
