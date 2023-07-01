from exception import NewsArticleSortingException
from logger import logging
from preprocessor import Preprocessor,Prediction
import pandas as pd
import os,sys
import streamlit as st

checkpoint = 'google/bert_uncased_L-4_H-256_A-4'   #bert-mini

model_path = 'model_files/'

st.title("News Article Sorting App")


try:
    logging.info("Reading the input data for prediction")
    st.header("Please enter/paste your text here")
    text = st.textarea("Entering text here")
    
    logging.info("Creating objects of processor and prediction class")
    preprocessor = Preprocessor(text)
    final_text = preprocessor.apply_preprocessing()

    prediction = Prediction(checkpoint, final_text,model_path)

    logging.info("Getting the prediction result")
    result = prediction.find_prediction()
    label = result[0]['label']
    score = result[0]['score']

    st.subheader("Prediction:")
   
    st.write("Category of the news text:",label)
    
    st.write('Accuracy of the prediction:',score)

    

except Exception as e:
    raise NewsArticleSortingException(e, sys)







