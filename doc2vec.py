# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 14:55:05 2017

@author: Vishnu
"""

from gensim.models import doc2vec
from collections import namedtuple
import numpy as np
import math
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text = wnl.lemmatize(letters_only_text)
    words = letters_only_text.lower().split()
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in words if w not in stopword_set]
    cleaned_word_list = " ".join(meaningful_words)
    return cleaned_word_list

def docvec(doc_list):
    doc = []
    for i in doc_list:
        doc.append(preprocess(i))
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(doc):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    model = doc2vec.Doc2Vec(docs, size = 300, window = 300, min_count = 1, workers = 4)
    X = model.docvecs[0]
    vec = [model.docvecs[i+1] for i in range(len(doc) - 1)]
    return X, vec

def similarity(vec, vectors):
    similarity_list = []
    for i in vectors: 
        cosine_similarity = np.dot(vec, i) / (np.linalg.norm(vec) * np.linalg.norm(i))
        similarity_list.append(cosine_similarity)
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
                similarity_list.append(cosine_similarity)
        except:
            cosine_similarity = 0
            similarity_list.append(cosine_similarity)
    return similarity_list

