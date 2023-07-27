from tkinter import *

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, log_loss

from sklearn.metrics import confusion_matrix

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


def show_the_Quality():

    new = np.array([[float(e1.get()),float(e2.get()),float(e3.get()),float(e4.get()),float(e5.get()),float(e6.get()),float(e7.get()),float(e8.get()),float(e9.get()),float(e10.get()),float(e11.get())]])

    Ans = RF_clf.predict(new)

    fin=str(Ans)[1:-1]

    quality.insert(0, fin)

data = pd.read_csv('wine__12.csv')

data.head()

data.describe()

extra = data[data.duplicated()]

extra.shape

y = data.quality

X = data.drop('quality', axis=1)

print(y.shape, X.shape)

colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('So , here is the correlation of different features----------->', y=1.05, size=15)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)

seed = 8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=seed)

RF_clf = RandomForestClassifier(random_state=seed)

cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')

RF_clf.fit(X_train, y_train)

pred_RF = RF_clf.predict(X_test)

master = Tk()

Label(master, text="Fixed Acidity-------------------------------------------------------->", anchor="nw", width=15).grid(row=0)

Label(master, text="Volatile Acidity----------------------------------------------------->", anchor="nw", width=15).grid(row=1)

Label(master, text="Citric Acid---------------------------------------------------------->", anchor="nw", width=15).grid(row=2)

Label(master, text="Residual Sugar------------------------------------------------------->", anchor="nw", width=15).grid(row=3)

Label(master, text="Chlorides------------------------------------------------------------>", anchor="nw", width=15).grid(row=4)

Label(master, text="Sulfur Dioxide------------------------------------------------------->", anchor="nw", width=15).grid(row=5)

Label(master, text="Total Sulfur Dioxide------------------------------------------------->", anchor="nw", width=15).grid(row=6)

Label(master, text="Density-------------------------------------------------------------->", anchor="nw", width=15).grid(row=7)

Label(master, text="pH------------------------------------------------------------------->", anchor="nw", width=15).grid(row=8)

Label(master, text="Sulphates------------------------------------------------------------>", anchor="nw", width=15).grid(row=9)

Label(master, text="Alcohol-------------------------------------------------------------->", anchor="nw", width=15).grid(row=10)

Label(master, text = "Quality------------------------------------------------------------->", anchor="nw", width=15).grid(row=13)

temp1 = Entry(master)

temp2 = Entry(master)

temp3 = Entry(master)

temp4 = Entry(master)

temp5 = Entry(master)

temp6 = Entry(master)

temp7 = Entry(master)

temp8 = Entry(master)

temp9 = Entry(master)

temp10 = Entry(master)

temp11 = Entry(master)

quality = Entry(master)

temp1.grid(row=0, column=1)

temp2.grid(row=1, column=1)

temp3.grid(row=2, column=1)

temp4.grid(row=3, column=1)

temp5.grid(row=4, column=1)

temp6.grid(row=5, column=1)

temp7.grid(row=6, column=1)

temp8.grid(row=7, column=1)

temp9.grid(row=8, column=1)

temp10.grid(row=9, column=1)

temp11.grid(row=10, column=1)

quality.grid(row=13, column=1)

Button(master, text='Quit', command=master.destroy,width=15).grid(row=11, column=0, sticky=W, pady=4)

Button(master, text='Find Quality', command=show_the_Quality,width=17).grid(row=11, column=1, sticky=W, pady=4)

mainloop( )