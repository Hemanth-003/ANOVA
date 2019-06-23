from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from plotly.offline import plot
from plotly.graph_objs import Scatter
from flask import Markup
# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)


@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()

    # read iris data set
    data = pd.read_csv('./database.csv')
    data.isnull().sum()
    data2 = pd.get_dummies(data, columns =[ 'Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude'] )
    X = data2.iloc[:,1:]
    y = data2.iloc[:,0]
    C=int(parameters['C'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=C)
    X_train.shape
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # fit model
    y_pred = gnb.predict(X_test)
    # persist model
    joblib.dump(gnb, 'model.pkl')

    return jsonify({'accuracy':round(gnb.score(X_test, y_test)*100,2) })


@app.route('/api/predict', methods=['POST'])
def predict():
    # get iris object from request
    X = request.get_json()
    X = [[float(X['Latitude']), float(X['Longitude'])]]
  

                           

 


if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
