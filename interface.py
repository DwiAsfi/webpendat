import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("APLIKASI PENAMBANGAN DATA")
st.write("By: Dwi Asfi Fajrin (200411100121)")
st.write("=========================================================================")
upload_data, preporcessing, modeling, implementation = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])


st.write("")


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah breast cancer dataset yang diambil dari https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
