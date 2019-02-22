# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 15:42:04 2017

@author: Vishnu
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()

def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text = wnl.lemmatize(letters_only_text)
    words = letters_only_text.lower().split()
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in words if w not in stopword_set]
    cleaned_word_list = " ".join(meaningful_words)
    return cleaned_word_list

def similarity(doc_list):
    doc = []
    for i in doc_list:
        doc.append(preprocess(i))
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return similarity[0]
