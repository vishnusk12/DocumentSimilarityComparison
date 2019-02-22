# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 11:22:33 2017

@author: Vishnu
"""

import gensim.models
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pyemd

wnl = WordNetLemmatizer()

path = 'C:/Users/hp/Downloads/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text = wnl.lemmatize(letters_only_text)
    words = letters_only_text.lower().split()
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in words if w not in stopword_set]
    cleaned_word_list = " ".join(meaningful_words)
    return cleaned_word_list

def similarity(user_input, doc_list):
    text1 = preprocess(user_input)
    similarity_list = []
    for i in doc_list:
        text2 = preprocess(i)
        distance = model.wmdistance(text1.split(), text2.split())
        similarity_list.append(distance)
    return similarity_list
