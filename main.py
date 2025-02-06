from streamlit.runtime.scriptrunner import add_script_run_ctx
add_script_run_ctx()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_ind
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import tensorboard
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARNING logs

# Streamlit title and description
st.title("AI & Data Science Project Dashboard")
st.write("### Interactive analysis of shopping and walking data")

# Load data directly from GitHub
DATA_URL = "https://raw.githubusercontent.com/Manish-Kumar-Rai/SM_AI/main/SM_AI_Project_Data_Updated.xlsx"

@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
    try:
        shopping_df = pd.read_excel(DATA_URL, sheet_name="Shopping Data")
        walking_df = pd.read_excel(DATA_URL, sheet_name="Walking Data")
        return shopping_df, walking_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

shopping_df, walking_df = load_data()

if shopping_df is not None and walking_df is not None:
    st.success("Data successfully loaded from GitHub repository!")

    # --- section to display raw data tables ---
    st.write("## Raw Data Preview")
    
    with st.expander("Show Shopping Data"):
        st.write("### Shopping Data Table")
        st.dataframe(shopping_df.head(10))  # Show first 10 rows
    
    with st.expander("Show Walking Data"):
        st.write("### Walking Data Table")
        st.dataframe(walking_df.head(10))  # Show first 10 rows
    
    # Data preprocessing
    shopping_df['Gender'] = shopping_df['Gender'].map({'M': 0, 'F': 1}).astype(int)
    shopping_df = pd.get_dummies(shopping_df, columns=['District'], drop_first=True).astype(float)
    
    # Linear Regression (Spending Prediction)
    X = shopping_df.drop(columns=['Spending (PLN)'])
    y = shopping_df['Spending (PLN)']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.write("### Linear Regression Results")
    st.text(model.summary())
    
    # Visualization: Spending by Gender
    st.write("### Spending Distribution by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x=shopping_df['Gender'], y=shopping_df['Spending (PLN)'], ax=ax)
    ax.set_xticklabels(["Male", "Female"])
    st.pyplot(fig)
    
    # Logistic Regression (Predict Gender)
    X_logit = shopping_df[['Spending (PLN)']]
    y_logit = shopping_df['Gender']
    logit_model = LogisticRegression().fit(X_logit, y_logit)
    accuracy = accuracy_score(y_logit, logit_model.predict(X_logit))
    st.write(f"### Logistic Regression Accuracy: {accuracy:.2f}")
    
    # K-Means Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(shopping_df[['Spending (PLN)', 'Weekday/Weekend']])
    kmeans = KMeans(n_clusters=2).fit(X_scaled)
    shopping_df['Cluster'] = kmeans.labels_
    
    # Visualization: Clusters
    st.write("### Customer Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=shopping_df['Spending (PLN)'], y=shopping_df['Weekday/Weekend'], hue=shopping_df['Cluster'], palette='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Hypothesis Testing
    men = shopping_df[shopping_df['Gender'] == 0]['Spending (PLN)']
    women = shopping_df[shopping_df['Gender'] == 1]['Spending (PLN)']
    t_stat, p_value = ttest_ind(men, women)
    st.write(f"### Hypothesis Testing p-value: {p_value:.4f}")
    
    # Walking Data Analysis
    st.write("## Walking Data Analysis")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    X_walk = walking_df[days]
    y_walk = walking_df['Total Calories Burned']
    X_walk = sm.add_constant(X_walk)
    model_walk = sm.OLS(y_walk, X_walk).fit()
    st.text(model_walk.summary())
    
    # Visualization: Calories Burned
    st.write("### Total Calories Burned Distribution")
    fig, ax = plt.subplots()
    sns.histplot(walking_df['Total Calories Burned'], kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    
    # Logistic Regression (Predict Gender from Walking Data)
    y_walk_logit = walking_df['Gender'].map({'M': 0, 'F': 1})
    logit_walk_model = LogisticRegression().fit(X_walk, y_walk_logit)
    accuracy_walk = logit_walk_model.score(X_walk, y_walk_logit)
    st.write(f"### Walking Data Logistic Regression Accuracy: {accuracy_walk:.2f}")
    
    # TensorBoard Logging
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    st.write("### TensorBoard Logging Initialized")
    st.write(f"Log directory: {log_dir}")
