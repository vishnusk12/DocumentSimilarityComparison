# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 14:36:13 2017

@author: Vishnu
"""

import gensim.models
import numpy as np
import math
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

path = 'C:/Users/hp/Downloads/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

class PhraseVector:
    def __init__(self, phrase):
        self.vector = self.PhraseToVec(phrase)
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
        if len(ignore) == 0:
            return np.mean(vectorSet, axis = 0)
        else:
            return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)
    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector=model[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity
    
def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text = wnl.lemmatize(letters_only_text)
    words = letters_only_text.lower().split()
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in words if w not in stopword_set]
    cleaned_word_list = " ".join(meaningful_words)
    return cleaned_word_list
    
def similarity(userInput1, doc_list):
    text1 = preprocess(userInput1)
    similarity_list = []
    for i in doc_list:
        text2 = preprocess(i)
        phraseVector1 = PhraseVector(text1)
        phraseVector2 = PhraseVector(text2)
        similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
        similarity_list.append(similarityScore)
    return similarity_list
