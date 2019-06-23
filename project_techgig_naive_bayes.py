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
data = pd.read_csv('C:\Kabirs Worspace\Personal\Data Science\Anova\database.csv')
#data.head()
data = data.dropna()
print(data.shape)
print(list(data.columns))

# BAR PLOT  for our required data can be obtained with below syntax :

"""sns.countplot(x='Location Source',data=data, palette='hls')
sns.countplot(x='Magnitude Source',data=data, palette='hls') """

sns.countplot(x='Horizontal Error',data=data, palette='hls')


plt.show()

# --------->    Checking null values:

data.isnull().sum()

data2 = pd.get_dummies(data, columns =['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error', 'Depth Seismic Stations', 'Magnitude', 'Magnitude Type', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'] )


#---------->    To drop unknown coloumns 

sns.heatmap(data2.corr())
plt.show()


# To find correlation

#---------->    Train the model using the training sets


X = data2.iloc[:,1:]
y = data2.iloc[:,0]


#---------->    Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape


#---------->    It shows Training data after splitting

#Create a Gaussian Classifier

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)



# ---------> Model Accuracy, how often is the classifier correct?

print('Accuracy of Naive_bayes classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))



""" ANOTHER METHOD for accuracy checking

from sklearn import metrics
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) """


