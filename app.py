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
keywords = joblib.load("models/keywords.pkl")


# Title

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write(
    "This application predicts the difficulty level and rating of a "
    "programming problem based on its textual description."
)

st.divider()

# Input Section

st.markdown("### 🔹 Enter Problem Details")

problem_text = st.text_area(
    "Enter the full problem description",
    placeholder="Paste the problem statement here...",
    height=250
)

st.divider()

def keyword_frequency(text):
    text = text.lower()
    return [text.count(k) for k in keywords]

# Prediction

if st.button("🔍 Predict Difficulty"):
    text = problem_text.strip()

    if not text:
        st.warning("Please enter a problem description.")
        st.stop()

    X_tfidf = tfidf.transform([text])
    kw_features = np.array([keyword_frequency(text)])

    X_input = np.hstack([X_tfidf.toarray(), kw_features])

    
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

    st.caption("Prediction is based on the problem's textual description.")

st.divider()

