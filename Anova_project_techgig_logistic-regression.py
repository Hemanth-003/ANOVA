import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt



plt.rc("font", size=14)

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn import train_test_split()

import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('data.csv', header=0)
data.head()
data = data.dropna()
print(data.shape)
print(list(data.columns))

#Bar plot:

sns.countplot(x='y',data=data, palette='hls')
plt.show()

#Checking null values:

data.isnull().sum()

data2 = pd.get_dummies(data, columns =['column1','column2'] )


#Manaki telini columns drop cheyyadaniki


sns.heatmap(data2.corr())
plt.show()


#Screen shot cut ayyipoindi annav ga aa data ki visualistaion....to find correlation


X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Splliting of data into training and test data


X_train.shape


#Training data entha undi ani chupistadi split ayyina tarawatha


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#Fiiting in logistic regression


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#Test set result predict cheyyadaniki


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
