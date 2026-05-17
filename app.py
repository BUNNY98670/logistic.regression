import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Title
st.title("Logistic Regression Prediction App")

st.write("Enter input value to predict PASS or FAIL")

# Input from user
x = st.number_input("Enter value of X", min_value=0.0)

# Dummy training data (you can replace with your dataset)
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict button
if st.button("Predict"):
    # Prediction
    prediction = model.predict([[x]])
    probability = model.predict_proba([[x]])[0][1]

    st.write("Probability:", probability)

    if probability >= 0.6:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

    # Show predicted values
    st.write("Predicted Value:", prediction[0])

# Extra: Show dataset
if st.checkbox("Show Training Data"):
    df = pd.DataFrame({"X": X.flatten(), "Target": y})
    st.write(df)