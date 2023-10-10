# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""

import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np



# Title
st.header("Adaboost+NearMiss ACLF death prediction model")

#input
ast=st.sidebar.number_input("ast(IU/L,Norm:10-40)")
bilirubin=st.sidebar.number_input("Total bilirubin(mg/dL,Norm:0.3-1.9)")
INR=st.sidebar.number_input("INR(Norm:0.8-1.2)")
WBC=st.sidebar.number_input("WBC(K/μL,Norm:4.5-11.5)")
platelet_count=st.sidebar.number_input("platelet count(K/μL,Norm:150-450)")
creatinine=st.sidebar.number_input("creatinine(mg/dL,Norm:0.5-1.2)")
sodium=st.sidebar.number_input("sodium(mEq/L,Nrom:135-145)")
heart_rate=st.sidebar.number_input("heart rate(bpm,Norm:60-100)")
dbp=st.sidebar.number_input("dbp(mmHg,Norm:60-80)")
temperature=st.sidebar.number_input("temperature(°C,Norm:36.5-37.5)")
spo2=st.sidebar.number_input("spo2(%,Norm:95-100)")
age=st.sidebar.number_input("age(year)")



with open('Adaboost+NearMiss_12.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('data_max_12.pkl', 'rb') as f:
    data_max = pickle.load(f)
with open('data_min_12.pkl', 'rb') as f:
    data_min = pickle.load(f)
with open('explainer_12.pkl', 'rb') as f:
    explainer = pickle.load(f)


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    # Store inputs into dataframe
    columns = ['INR','creatinine','bilirubin','WBC','sodium','platelet_count','temperature','dbp','ast','spo2','heart_rate','age']
    X = pd.DataFrame([[INR,creatinine,bilirubin,WBC,sodium,platelet_count,temperature,dbp,ast,spo2,heart_rate,age]], 
                     columns =columns )
    st.write('Raw data:')
    st.dataframe(X)
    X = (X-data_min)/(data_max-data_min)
    st.write('Normalized data:')
    st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X)
    pred=clf.predict_proba(X)[0][1]
    shap_values2 = explainer(X)
    
    # Output prediction
    
    st.text(f"The probability of death of the patient is {pred}.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig=shap.plots.bar(shap_values2[0])
    st.pyplot(fig)
    
    
    
    
    
    
    
