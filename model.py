# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 19:35:00 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:31:14 2023

@author: Administrator
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Building the model
df = pd.read_csv(r"C:\Users\Administrator\Desktop\variables\Project\iris.csv")
df.head()
df.describe()
df.info()
df['variety'].value_counts()
df.isnull().sum()

df['sepal.length'].hist()
df['sepal.width'].hist()
df['petal.length'].hist()
df['petal.width'].hist()
colors = ['pink', 'brown', 'blue']
variety = ['Iris-virginica','Iris-versicolor','Iris-setosa']

for i in range(3):
    x = df[df['variety'] == variety[i]]
    plt.scatter(x['sepal.length'], x['sepal.width'], c = colors[i], label=variety[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

for i in range(3):
    x = df[df['variety'] == variety[i]]
    plt.scatter(x['petal.length'], x['petal.width'], c = colors[i], label=variety[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

for i in range(3):
    x = df[df['variety'] == variety[i]]
    plt.scatter(x['sepal.length'], x['petal.length'], c = colors[i], label=variety[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()

for i in range(3):
    x = df[df['variety'] == variety[i]]
    plt.scatter(x['sepal.width'], x['petal.width'], c = colors[i], label=variety[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()

df.corr()
corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')

#Training the model
from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['variety'])
Y = df['variety']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)

print("Accuracy: ",model.score(x_test, y_test) * 100)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)

# Saving model to disk
import pickle
filename = 'savedmodel.pkl'
pickle.dump(model, open(filename, 'wb'))
# Loading model to compare the results
x_test.head()
load_model = pickle.load(open(filename,'rb'))
load_model.predict([[6.0, 2.2, 4.0, 1.0]])
