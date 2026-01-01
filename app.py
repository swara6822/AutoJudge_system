import streamlit as st
import numpy as np
import joblib

# Load trained models

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# App UI

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write(
    "Enter problem metadata below to predict the difficulty level "
    "and estimated difficulty rating."
)

# User Inputs

tag1 = st.text_input("Problem Tag 1", "")
tag2 = st.text_input("Problem Tag 2", "")
tag3 = st.text_input("Problem Tag 3", "")
tag4 = st.text_input("Problem Tag 4", "")

solved = st.number_input(
    "Number of Users Who Solved the Problem",
    min_value=0,
    value=1000,
    step=1
)

# Prediction

if st.button("Predict Difficulty"):
    # Combine tags
    tags_text = f"{tag1} {tag2} {tag3} {tag4}".strip()

    # TF-IDF transform
    X_tags = tfidf.transform([tags_text])

    # Combine with solved count
    X_input = np.hstack((X_tags.toarray(), [[solved]]))

    # Classification prediction
    class_pred_encoded = clf.predict(X_input)
    class_pred = label_encoder.inverse_transform(class_pred_encoded)[0]

    # Regression prediction
    rating_pred = reg.predict(X_input)[0]

    # Display Results

    st.success("Prediction Successful!")

    st.markdown(f"### Predicted Difficulty Level: **{class_pred}**")
    st.markdown(f"### Predicted Difficulty Rating: **{int(rating_pred)}**")
