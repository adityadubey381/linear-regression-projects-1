import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px

#Data
df = pd.read_csv(r"C:\Users\Aditya kumar Dubey\OneDrive\Documents\Desktop\Data\SLR_placement.csv")

st.title("This is my first Linear regression project.")

ch = st.sidebar.radio(
    'Select an Option',
    ('Data Graphical View','Salary Predictor','About me')
)


# ML Concept

X = df.iloc[:,0:1]
Y = df.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)




if ch == 'Data Graphical View':
    st.subheader(" Scatter plot")
    fig, ax = plt.subplots()
    ax = plt.scatter(df['cgpa'], df['package'])
    plt.title('Scatter Plot of CGPA vs. Package')
    plt.xlabel('CGPA')
    plt.ylabel('Package')
    st.pyplot(fig)



    st.subheader("Drawing a Best fit line on Scatter Plot")
    fig, ax = plt.subplots()
    ax =plt.scatter(df['cgpa'],df['package'])
    plt.plot(X_train,lr.predict(X_train), color = 'red')
    plt.title('Scatter Plot of CGPA vs. Package')
    plt.xlabel('CGPA')
    plt.ylabel('Package')
    st.pyplot(fig)


if  ch =='Salary Predictor':
    lr.predict(X_test.iloc[2].values.reshape(1, 1))
    m = lr.coef_
    b = lr.intercept_
    # Taking a numeric input
    cgpa = st.number_input("Enter your CGPA:", value=0.0)
    salary = m * cgpa + b
    st.write(salary)

if ch =='About me':
    st.markdown("""
        [Click here to visit my LinkedIn profile](https://www.linkedin.com/in/aditya-kumar-dubey-9833b4278/)
    """)
