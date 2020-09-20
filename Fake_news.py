# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:33:02 2020

@author: User
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
df = pd.read_csv("train.csv")
conversion_dict = {0: 'Real' , 1: 'Fake'}
df['label'] = df['label'].replace(conversion_dict)
print(df.label.value_counts())
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75)
vec_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test=tfidf_vectorizer.transform(x_test.values.astype('U'))
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)
y_pred=pac.predict(vec_test)
score=accuracy_score(y_test,y_pred)
print(f'PAC Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred, labels=['Real','Fake'])
X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
scores = cross_val_score(pac, X, df['label'].values, cv=5)
print(f'K Fold Accuracy: {round(scores.mean()*100,2)}%')


