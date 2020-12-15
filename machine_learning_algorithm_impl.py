import pandas as pd
import preprocessing as pp
from sklearn import model_selection,preprocessing
import bag_of_words as bow
import tf_idf as tfi
import modeling as md
import dtree as dt
import knn as k
import Svm as s
import SGD_Svm as sg
import Naive_Bayes as nb
import Gaussian_Naive_Bayes as gnb
import Bernoulli_Naive_Bayes as bnb
train_data=pd.read_csv('nfr.csv')
train_data=train_data[pd.notnull(train_data['Class'])]
print(train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
my_tags = ['A','F','FT','L','LF','MN','O','PE','SC','SE','US']
train_data=pp.preprocessing(train_data)
print(train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())

# split the dataset into training and validation datasets
X = train_data.RequirementText_string
y = train_data.Class
train_data['text']=X
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X,y,test_size=0.15, random_state = 0)
print (train_x.shape)
print (valid_x.shape)
print (train_y.shape)
print (valid_y.shape)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#Count vectorizing the training data
train=[]
train=bow.bag_of_words(train_data,X,train_x, valid_x)
xtrain_count=train[0]
xvalid_count=train[1]
#Tf-IDF of training data
tf_idf=[]
tf_idf=tfi.tf_idf(train_data,X,train_x, valid_x)
xtrain_tfidf=tf_idf[0]
xvalid_tfidf=tf_idf[1]
xtrain_tfidf_ngram=tf_idf[2]
xvalid_tfidf_ngram=tf_idf[3]
xtrain_tfidf_ngram_chars=tf_idf[4]
xvalid_tfidf_ngram_chars=tf_idf[5]

#Naive Bayes classifier implementation
nb.Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#Bernoulli Naive Bayes classifier implementation
bnb.Bernoulli_Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#Gaussian Naive Bayes classifier implementation
gnb.Gaussian_Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#Decision tree classifier implementation
dt.dtree(train,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#KNN classifier implementation
k.knn(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#Linear SVM classifier implementation
s.Svm(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)
#SGD SVM classifier implementation
sg.SGD_Svm(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)

