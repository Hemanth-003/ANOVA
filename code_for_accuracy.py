

# Importing predefined libraries required

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)

from sklearn.model_selection import cross_validate

# --------->  Using NAIVE_BAYES CLASSIFIER 

from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#Here database.csv is the data on which basis we are going to predict. Make sure that both the files are in the same folder for succesful execution.

# Replace the file path below based on where the file is placed before running the code
data = pd.read_csv('database.csv')
#print(data)
#data.head()
#data = data.dropna()


data_clean=data.drop(['Depth Error','Depth Seismic Stations','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap','Horizontal Distance','Horizontal Error','Root Mean Square'],axis=1)
data_clean=data_clean.dropna()
data_clean=data_clean.drop(['Date','Time','Type'],axis=1)
data_clean=data_clean.drop(['ID','Source','Location Source','Magnitude Source','Status'],axis=1)

#data_clean[data_clean[0::,0].astype(np.float) >= 'Magnitude Type', 5] = 'Magnitude Type'-1.0
print(data_clean.shape)

X = data_clean.iloc[:,1:4]
y = data_clean.iloc[:,-2]

X=X.astype('float')
y=y.astype('str')
#---------->    Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape

#---------->    It shows Training data after splitting

#Create a Gaussian Classifier

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)



# ---------> Model Accuracy, how often is the classifier correct?

print('Accuracy of Naive_bayes classifier on test set: {:.2f}\n\n'.format(gnb.score(X_train, y_train)))



# ANOTHER METHOD for accuracy checking

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


