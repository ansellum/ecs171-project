from sklearn import svm
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
    cap_shape = st.sidebar.selectbox(
        'Select a cap shape',
        options=['b','c','x','f','k',', s'],
		help="bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s")
    cap_surface = st.sidebar.selectbox(
        'Select a cap surface',
        options=['f','g','y','s'],
		help="fibrous=f,grooves=g, scaly=y, smooth=s")
    cap_color = st.sidebar.selectbox(
        'Select a cap color',
        options=['n','b','c','g','r','p','u','e','w','y'],
		help="brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y")
    bruises = st.sidebar.selectbox(
        'Select bruises',
        options=['t','f'],
		help="true=t, false=f")
    odor = st.sidebar.selectbox(
        'Select an odor',
        options=['a','l','c','y','f','m','n','p','s'],
		help="almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s")
    gill_attachment = st.sidebar.selectbox(
        'Select a gill attachment',
        options=['a','f'],
		help="attached=a, free=f") # removed 2
    gill_spacing = st.sidebar.selectbox(
        'Select a gill spacing',
        options=['c','w'],
		help="close=c, crowded=w,") # removed d
    gill_size = st.sidebar.selectbox(
        'Select a gill_size',
        options=['b','n'],
		help="broad=b, narrow=n")
    gill_color = st.sidebar.selectbox(
        'Select a gill color',
        options=['k','n','b','h','g','r','o','p','u','e','w','y'],
		help="black=k, brown=n, buff=b, chocolate=h, gray=g,  green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y")
    stalk_shape = st.sidebar.selectbox(
        'Select a stalk shape',
        options=['e','t'],
		help="enlarging=e, tapering=t")
    stalk_root = st.sidebar.selectbox(
        'Select a stalk root',
        options=['e','c','b','r','?'],
		help="bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?")
    stalk_surface_above_ring = st.sidebar.selectbox(
        'Select a stalk surface above ring',
        options=['f','y','k','s'],
		help="fibrous=f, scaly=y, silky=k, smooth=s")
    stalk_surface_below_ring = st.sidebar.selectbox(
        'Select a stalk surface below ring',
        options=['f','y','k','s'],
		help="fibrous=f, scaly=y, silky=k, smooth=s")
    stalk_color_above_ring = st.sidebar.selectbox(
        'Select a stalk color above ring',
        options=['n','b','c','g','o','p','e','w','y'],
		help="brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y")
    stalk_color_below_ring = st.sidebar.selectbox(
        'Select a stalk color below ring',
        options=['n','b','c','g','o','p','e','w','y'],
		help="brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y")
    veil_color = st.sidebar.selectbox(
        'Select a veil color',
        options=['n','o','w','y'],
		help="brown=n, orange=o, white=w, yellow=y")
    ring_number = st.sidebar.selectbox(
        'Select a ring number',
        options=['n','o','t'],
		help="none=n, one=o, two=t")
    ring_type = st.sidebar.selectbox(
        'Select a ring type',
        options=['p','e','l','f','n'],
		help="evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z")  # Removed 'c' not in training data
    spore_print_color = st.sidebar.selectbox(
        'Select a spore print color',
        options=['k','n','b','h','r','o','u','w','y'],
		help="black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y")
    population = st.sidebar.selectbox(
        'Select a population',
        options=['a','c','n','s','v','y'],
		help="abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y")
    habitat = st.sidebar.selectbox(
        'Select a habitat',
        options=['g','l','m','p','u','w','d'],
		help="grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d")
    data = {'cap-shape': cap_shape,
           'cap-surface': cap_surface,
            'cap-color': cap_color,
            'bruises': bruises,
            'odor': odor,
            'gill-attachment': gill_attachment,
            'gill-spacing': gill_spacing,
            'gill-size': gill_size,
            'gill-color': gill_color,
            'stalk-shape': stalk_shape,
            'stalk-root': stalk_root,
            'stalk-surface-above-ring': stalk_surface_above_ring,
            'stalk-surface-below-ring': stalk_surface_below_ring,
            'stalk-color-above-ring': stalk_color_above_ring,
            'stalk-color-below-ring': stalk_color_below_ring,
            'veil-color': veil_color,
            'ring-number': ring_number,
            'ring-type': ring_type,
            'spore-print-color': spore_print_color,
            'population': population,
            'habitat': habitat
           }
    features = pd.DataFrame(data, index=[0])
    return features
    
    
user_input = user_input_features()

st.subheader('User Input Parameters')
st.write(user_input)

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
	df_shuffled = df.sample(frac=1)
	# split into input (X) and output (y) variables & convert to numPy array
	X = df_shuffled.drop(column, axis = 1)
	y = df_shuffled[column]
	return X, y

# prepare input data
def prepare_inputs(X):
	oe = OrdinalEncoder()
	oe.fit(X)
	X_enc = oe.transform(X)
		
	# scale dataset
	scaler = preprocessing.MinMaxScaler()
	X_enc_scaled = scaler.fit_transform(X_enc)
	return X_enc_scaled, oe, scaler

# prepare target
def prepare_targets(y):
	le = LabelEncoder()
	le.fit(y)
	y_enc = le.transform(y)
	return y_enc

# load the dataset
X, y = load_dataset('mushrooms.csv', 'class')
X = X.drop('veil-type', axis = 1)

# prepare input data
X_enc, xoe, xscaler = prepare_inputs(X)
# prepare output data
y_enc = prepare_targets(y)

# split into train and test sets; 80/20 split
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_enc, y_enc, test_size=0.2, random_state=1)

rbf = svm.SVC(kernel='rbf', gamma=0.5).fit(X_train_enc, y_train_enc)

rbf.fit(X_train_enc, y_train_enc)
# pred = mlp.predict(X_test_enc)

input_pre = prepare_new_input(user_input, xoe, xscaler)
pred = rbf.predict(input_pre)

# st.subleader('Class labels and their corresponding index number')
# st.write(df2.target_names)
st.subheader('Prediction')
st.write(pred)
