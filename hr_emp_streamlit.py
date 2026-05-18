import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title='HR Attrition Prediction', layout='wide')
st.title('HR Employee Attrition Prediction App')

uploaded_file = st.file_uploader('Upload HR-Employee-Attrition.csv', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    st.subheader('Null Values')
    st.write(df.isnull().sum())

    for col in df.select_dtypes(include='object'):
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include=np.number):
        df[col].fillna(df[col].median(), inplace=True)

    st.subheader('Bar Graph')
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader('Scatter Plot')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='MonthlyIncome', hue='Attrition', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader('Boxplot')
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax)
    st.pyplot(fig)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.subheader('Model Accuracy')
    st.write(acc)

    st.subheader('Confusion Matrix')
    st.write(cm)

    st.subheader('Classification Report')
    st.text(report)

    st.success('Model Trained Successfully!')
else:
    st.info('Upload CSV file to begin.')