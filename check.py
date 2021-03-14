import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(r'G:\web_dev\movie_recommendation\AJAX-Movie-Recommendation-System-with-Sentiment-Analysis-master\datasets\main_data.csv')
    # creating a count matrix

cv=CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])
#print(count_matrix)
print(count_matrix.shape)
    # creating a similarity score matrix
similarity = cosine_similarity(count_matrix)

m="liar liar"
m = m.lower()
if m not in data['movie_title'].unique():
        print('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
else:
        i = m.index()
        print(similarity[i])
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
print(i) 
print(lst)
print(l)           
