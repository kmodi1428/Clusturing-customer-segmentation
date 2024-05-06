import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# cleaned and preprocessed dataset
df= pd.read_csv('data.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])

st.set_page_config(page_title= 'Project-p379', page_icon='🤡')
st.title("User Behaviour showing in cluster")

with st.sidebar:
    Education= st.radio("Select Education", ['Under Graduate', 'Post Graduate'])
    Marital_Status= st.radio("Select Marital", ['Single','Relationship',])
    Income= st.slider("Select income", 0, 100000)
    Kids= st.radio("no. kids", ['0', '1', '2', '3', '4'])
    Expenses= st.slider("Select the Expenses", 0, 10000)
    TotalAcceptedCmp= st.slider("select cmp_accpetance", 0, 5)
    Customer_Age= st.slider("Select Age", 1, 100)
    submit= st.button("Submit")
    #show= st.button("Show")

if submit:
    user_data = {'Marital_Status': [Marital_Status], 'Education': [Education], 'Kids':[Kids], 'TotalAcceptedCmp':[TotalAcceptedCmp], 'Expenses':[Expenses], 'Income':[Income], 'Customer_Age':[Customer_Age]}
    user_df = pd.DataFrame(user_data)

    # Preprocessing (convert categorical variables to numerical)
    user_df['Education'] = user_df['Education'].map({'Under Graduate': 0, 'Post Graduate': 1})
    user_df['Marital_Status'] = user_df['Marital_Status'].map({'Single': 0, 'Relationship': 1})
    user_df['Kids'] = user_df['Kids'].astype(int)

    # Select features for clustering
    features = ['Education', 'Marital_Status', 'Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 'Customer_Age']

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(df[features])

    # Extract user cluster
    user_cluster = kmeans.predict(user_df[features])[0]

    #if show:
        # Plotting
        #fig, ax = plt.subplots(figsize=(12, 8))
        #sns.scatterplot(data=df, x='Income', y='Expenses', hue=clusters, palette='rainbow', ax=ax)
        #ax.scatter(Income, Expenses, c='red', marker='*', s=200, label='Your Position')
        #ax.set_title('KMeans Clustering')
        #ax.set_xlabel('Income')
        #ax.set_ylabel('Expenses')
        #ax.legend()
        #st.pyplot(fig)

    # Display user cluster
    st.write(f'Your cluster: {user_cluster}') 