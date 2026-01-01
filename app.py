import streamlit as st
import numpy as np
import joblib

# Page config

st.set_page_config(
    page_title="AutoJudge",
    layout="centered"
)

# Custom CSS (fix red input highlight)

st.markdown(
    """
    <style>
    input, textarea {
        border: 1px solid #d0d0d0 !important;
        box-shadow: none !important;
    }
    input:focus, textarea:focus {
        border: 1px solid #a0a0a0 !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained models

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Title

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write(
    "This application predicts the difficulty level and rating of a "
    "programming problem based on its tags and solve statistics."
)

st.divider()

# Input Section

st.markdown("### 🔹 Enter Problem Details")

tag1 = st.text_input("Problem Tag 1", placeholder="e.g. dp")
tag2 = st.text_input("Problem Tag 2", placeholder="e.g. greedy")
tag3 = st.text_input("Problem Tag 3", placeholder="e.g. graphs")
tag4 = st.text_input("Problem Tag 4", placeholder="optional")

solved = st.number_input(
    "Number of users who solved the problem",
    min_value=0,
    value=3000,
    step=100
)

st.divider()

# Prediction

if st.button("🔍 Predict Difficulty"):
    tags_text = f"{tag1} {tag2} {tag3} {tag4}".strip()

    X_tags = tfidf.transform([tags_text])
    X_input = np.hstack((X_tags.toarray(), [[solved]]))

    class_pred_encoded = clf.predict(X_input)
    class_pred = label_encoder.inverse_transform(class_pred_encoded)[0]

    rating_pred = int(reg.predict(X_input)[0])

    st.markdown("### 📊 Prediction Result")

    # Color-coded difficulty
    if class_pred == "Easy":
        st.success(f"🟢 Difficulty Level: **{class_pred}**")
    elif class_pred == "Medium":
        st.warning(f"🟡 Difficulty Level: **{class_pred}**")
    else:
        st.error(f"🔴 Difficulty Level: **{class_pred}**")

    st.markdown(f"**Predicted Difficulty Rating:** `{rating_pred}`")

    st.caption("Prediction is based on problem tags and historical solve statistics.")

st.divider()

