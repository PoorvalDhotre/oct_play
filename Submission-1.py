#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#Import Data
test=pd.read_csv(r'C:\Users\poorv\Poorval\Analytics\git\oct_play\test.csv')
train=pd.read_csv(r'C:\Users\poorv\Poorval\Analytics\git\oct_play\train.csv')



#First submission running a random forest model without any feature engineering.


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