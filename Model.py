from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
# load the dataset
def load_dataset(filename, column):
	# load the dataset as a pandas DataFrame
	df = read_csv(filename)
	# split into input (X) and output (y) variables & convert to numPy array
	X = df.drop(column, axis = 1).values
	y = df[column].values
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
		
	# scale dataset
	scaler = preprocessing.MinMaxScaler()
	X_train_rescaled = scaler.fit_transform(X_train_enc)
	X_test_rescaled = scaler.fit_transform(X_test_enc)
	return X_train_rescaled, X_test_rescaled

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# load the dataset
X, y = load_dataset('mushrooms.csv', 'class')
# split into train and test sets; 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'logistic', learning_rate_init = 0.1, batch_size = 100, hidden_layer_sizes = (100,), max_iter = 200)

mlp.fit(X_train_enc, y_train_enc)
pred = mlp.predict(X_test_enc)
