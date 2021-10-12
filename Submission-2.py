#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#Import Data
test=pd.read_csv(r'C:\Users\poorv\Poorval\Analytics\Portfolio\Classification\Oct Tabular Playground\test.csv')
train=pd.read_csv(r'C:\Users\poorv\Poorval\Analytics\Portfolio\Classification\Oct Tabular Playground\train.csv')


#Features have been scaled. Look for values above 1 in the whole table
#Find feature pairs with >0.7 correlation and reduce dimensions into a single feature to reduce multicollinearity

#Checking for null values
train.isnull().sum()
test.isnull().sum()


#Checking if data is balanced
train['target'].value_counts()

#Bar Plot of target
plt.bar(train['target'].unique(),train['target'].value_counts())
sns.barplot(train['target'].unique(),train['target'].value_counts())

#Find Duplicate rows
train.iloc[:,1:][train.iloc[:,1:].duplicated()]










#Features and Target Split
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values


#Create and Fit Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10)

model.fit(x_train, y_train)

#Predicting Probabilities
proba=model.predict_proba(test)


#Predicting Categories
predicted=model.predict(test)

#Preparing Submission
submission=pd.concat([test['id'], pd.DataFrame(proba, columns=[1,'target'])['target']], axis=1)
submission.to_csv(r'C:\Users\poorv\Poorval\Analytics\Portfolio\Classification\Oct Tabular Playground\Submission\Submission1.csv',index=False)



