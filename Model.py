from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import streamlit as st
st.write("""
# Mushroom Toxicity Prediction

This app predicts whether a mushroom is toxic or edible!
""")
st.sidebar.header('User Input Parameters')

def user_input_features():
    cap_shape = st.sidebar.select_slider(
        'Select a cap shape',
        options=['b','c','x','f','k','s'])
    cap_surface = st.sidebar.select_slider(
        'Select a cap surface',
        options=['f','g','y','s'])
    cap_color = st.sidebar.select_slider(
        'Select a cap color',
        options=['n','b','c','g','r','p','u','e','w','y'])
    bruises = st.sidebar.select_slider(
        'Select bruises or no bruises',
        options=['t','f'])
    odor = st.sidebar.select_slider(
        'Select a odor',
        options=['a','l','c','y','f','m','n','p','s'])
    gill_attachment = st.sidebar.select_slider(
        'Select a gill attachment',
        options=['a','f']) # removed 2
    gill_spacing = st.sidebar.select_slider(
        'Select a gill spacing',
        options=['c','w']) # removed d
    gill_size = st.sidebar.select_slider(
        'Select a gill_size',
        options=['b','n'])
    gill_color = st.sidebar.select_slider(
        'Select a gill color',
        options=['k','n','b','h','g','r','o','p','u','e','w','y'])
    stalk_shape = st.sidebar.select_slider(
        'Select a stalk shape',
        options=['e','t'])
    stalk_root = st.sidebar.select_slider(
        'Select a stalk root',
        options=['e','c','b','r','?'])
    stalk_surface_above_ring = st.sidebar.select_slider(
        'Select a stalk surface above ring',
        options=['f','y','k','s'])
    stalk_surface_below_ring = st.sidebar.select_slider(
        'Select a stalk surface below ring',
        options=['f','y','k','s'])
    stalk_color_above_ring = st.sidebar.select_slider(
        'Select a stalk color above ring',
        options=['n','b','c','g','o','p','e','w','y'])
    stalk_color_below_ring = st.sidebar.select_slider(
        'Select a stalk color below ring',
        options=['n','b','c','g','o','p','e','w','y'])
    veil_color = st.sidebar.select_slider(
        'Select a veil color',
        options=['n','o','w','y'])
    ring_number = st.sidebar.select_slider(
        'Select a ring number',
        options=['n','o','t'])
    ring_type = st.sidebar.select_slider(
        'Select a ring type',
        options=['p','e','l','f','n'])  # Removed 'c' not in training data
    spore_print_color = st.sidebar.select_slider(
        'Select a spore print color',
        options=['k','n','b','h','r','o','u','w','y'])
    population = st.sidebar.select_slider(
        'Select a population',
        options=['a','c','n','s','v','y'])
    habitat = st.sidebar.select_slider(
        'Select a population',
        options=['g','l','m','p','u','w','d'])
    data = {'cap_shape': cap_shape,
           'cap_surface': cap_surface,
            'cap_color': cap_color,
            'bruises': bruises,
            'odor': odor,
            'gill_attachment': gill_attachment,
            'gill_spacing': gill_spacing,
            'gill_size': gill_size,
            'gill_color': gill_color,
            'stalk_shape': stalk_shape,
            'stalk_root': stalk_root,
            'stalk_surface_above_ring': stalk_surface_above_ring,
            'stalk_surface_below_ring': stalk_surface_below_ring,
            'stalk_color_above_ring': stalk_color_above_ring,
            'stalk_color_below_ring': stalk_color_below_ring,
            'veil_color': veil_color,
            'ring_number': ring_number,
            'ring_type': ring_type,
            'spore_print_color': spore_print_color,
            'population': population,
            'habitat': habitat
           }
    features = pd.DataFrame(data, index=[0])
    return features
    
    
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Function to prepare the new inputs
def prepare_new_input(df,oe,scaler):
    # Apply the ordinal encoder
    df_encoded = oe.transform(df)
    # Apply the min-max scaler
    df_rescaled = scaler.transform(df_encoded)
    return df_rescaled
# load the dataset
def load_dataset(filename, column):
    # load the dataset as a pandas DataFrame
    df = read_csv(filename)
    # Drop the 'veil_type' column
    df = df.drop('veil-type', axis=1)
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
    return X_train_rescaled, X_test_rescaled, oe, scaler

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
X_train_enc, X_test_enc, xoe, scaler = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'logistic', learning_rate_init = 0.1, batch_size = 100, hidden_layer_sizes = (100,), max_iter = 200)

mlp.fit(X_train_enc, y_train_enc)
# pred = mlp.predict(X_test_enc)

input_pre = prepare_new_input(df,xoe, scaler)
pred = mlp.predict(input_pre)
pred_proba = mlp.predict(input_pre)

# st.subleader('Class labels and their corresponding index number')
# st.write(df2.target_names)
st.subheader('Prediction')
st.write(pred)
st.subheader('Prediction Probability')
st.write(pred_proba)
