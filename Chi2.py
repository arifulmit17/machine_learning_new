# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:44:33 2020

@author: User
"""
import pandas as pd
import preprocessing as pp
from sklearn import preprocessing
from sklearn import model_selection

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',150)

train_data=pd.read_csv('nfr.csv')
train_data=train_data.drop("ProjectID", axis=1)
train_data=train_data[pd.notnull(train_data['Class'])]
print(train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
my_tags = ['A','F','FT','L','LF','MN','O','PE','SC','SE','US']
train_data=pp.preprocessing(train_data)
#print(train_data)
#my_tags=train_data['Class']
print(train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
X = train_data.RequirementText_string
y = train_data.Class
train_data['text']=X
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X,y,test_size=0.15, random_state = 0)
print (train_x.shape)
print (valid_x.shape)
print (train_y.shape)
print (valid_y.shape)
from TextFeatureSelection import TextFeatureSelection

#Multiclass classification problem
#input_doc_list=['i am very happy','i just had an awesome weekend','this is a very difficult terrain to trek. i wish i stayed back at home.','i just had lunch','Do you want chips?']
#target=['Positive','Positive','Negative','Neutral','Neutral']
#input_doc_list_2=train_data.drop("ProjectID", axis=1)
#input_doc_list_2=train_data
input_doc_list_2=train_y
#input_doc_list=train_data["RequirementText_string"]
#print(train_data["RequirementText_string"])
#print(train_x)
input_doc_list=train_x
input_doc_list=input_doc_list.values.tolist()
#print(input_doc_list)
target=input_doc_list_2
target=target.values.tolist()
#print(target)
fsOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list)
result=fsOBJ.getScore()
result=pd.DataFrame(data=result)
#print(result)
chi2=result["Chi Square"]
#print(result,file=open("result.txt", "a"))


#Binary classification
#input_doc_list=['i am content with this location','i am having the time of my life','you cannot learn machine learning without linear algebra','i want to go to mars']
#target=[1,1,0,1]
#fsOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list)
#result_df=fsOBJ.getScore()
print(chi2.shape)
fsOBJ=TextFeatureSelection(target=valid_y.values.tolist(),input_doc_list=valid_x.values.tolist())
result=fsOBJ.getScore()
result=pd.DataFrame(data=result)
#print(result)
chi2=result["Chi Square"]
print(chi2.shape)