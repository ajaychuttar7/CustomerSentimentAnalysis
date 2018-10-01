'''
we started with the coding on 29 sep
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1 = pd.read_csv('finaldatasettemp.csv', delimiter = ',',quoting = 3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0 , 1080):
    
    review = re.sub('[^a-zA-Z]',' ',dataset1['Customer Message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset1.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,dataset1['Category'],test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

print(classifier.predict(cv.transform(["please lower the price"])))

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
Rcm = confusion_matrix(y_test,y_pred)