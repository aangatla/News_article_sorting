from exception import NewsArticleSortingException
from logger import logging
import transformers
from transformers import pipeline
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt 
import seaborn as sns 
import re
import string,time
from textblob import TextBlob
import nltk
import os,sys
from nltk.corpus import stopwords
from transformers import AutoTokenizer,AutoConfig




nltk.download('stopwords')


    
class Preprocessor:
    try:
        def __init__(self, text):
            logging.info("Preprocessing starts")
            self.text = text
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

        
    try:
        def lowercase(self,text):
            logging.info("Converting the text into lowercase")
            return self.text.lower()
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

    
    try:
        
        def remove_html_tags(self,text):
            logging.info("Removing html tags from the text, if present")        
            pattern = re. compile(r'<[^>]+>')
            text = pattern.sub(r"", text)
            return text
    except Exception as e:
        raise NewsArticleSortingException(e, sys)


    try:

        def remove_url(self,text):
            logging.info("Removing url from the text, if present")        
            pattern = re.compile(r'http\S+')
            text = pattern.sub(r"", text)
            return text
    except Exception as e:
        raise NewsArticleSortingException(e, sys)
    
    try:
        def remove_punctuations(self,text):
            logging.info("Removing punctuations from the text, if present")    
            exclude = string.punctuation
            return text.translate(str.maketrans('', '', exclude))
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

        
    try:
        def spell_corrector(self,text):
            logging.info("Correcting the spelling of the words")    
            return TextBlob(text).correct().string
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

    try:
        def stop_words(self,text):
            logging.info("Removing the stopwords")
            new_text = []
            for word in text.split():
                if word in stopwords.words('english'):
                    new_text.append("")
                else:
                    new_text.append(word)
                
            return " ".join(new_text)
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

    try:        
        def apply_preprocessing(self):
            logging.info("Applying all the preprocessing steps")
            lowercase_text = self.lowercase(self.text)
            text_without_html = self.remove_html_tags(lowercase_text)
            text_without_url = self.remove_url(text_without_html)
            text_without_punctuations = self.remove_punctuations(text_without_url)
            correct_spelled_text = self.spell_corrector(text_without_punctuations)
            final_text = self.stop_words(correct_spelled_text)

            return final_text
    except Exception as e:
        raise NewsArticleSortingException(e, sys)
        

class Prediction:

    try:        
        def __init__(self,checkpoint,text,model_path):
            logging.info("Prediction")
            self.checkpoint = checkpoint
            self.text = text
            self.model_path = model_path
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

    try:
        def create_model(self,model_path):
            logging.info("Loading the tokenizer and model")
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            saved_model = pipeline('text-classification', model = self.model_path)
            return saved_model,tokenizer
    except Exception as e:
        raise NewsArticleSortingException(e, sys)        

    try:
        def predict(self,text,tokenizer,saved_model):
            logging.info("Doing the prediction")
            ids = tokenizer(self.text,max_length = 512, truncation = True)
            processed_text = tokenizer.decode(ids['input_ids'][1:-1])
            result = saved_model(processed_text)
            return result
    except Exception as e:
        raise NewsArticleSortingException(e, sys)

    try:    
        def find_prediction(self):
            saved_model,tokenizer = self.create_model(self.model_path)
            final_result = self.predict(self.text, tokenizer, saved_model)
            return final_result
    except Exception as e:
        raise NewsArticleSortingException(e, sys)        








                




    
