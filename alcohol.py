import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL ----------------
with open("student_alcohol_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("student_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Weekend Alcohol Predictor",
    page_icon="🍷",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #ffc0cb, #ffffff);
    color: #00008b;
}
div.stButton > button {
    background-color: #ff69b4;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🍷 Student Weekend Alcohol Consumption Predictor")
st.markdown(
    "Predict **weekend alcohol consumption (Walc)** using a few important student factors."
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Student Information")

def user_input_features():
    school = st.sidebar.selectbox("School", ["GP", "MS"])
    sex = st.sidebar.selectbox("Sex", ["F", "M"])
    age = st.sidebar.slider("Age", 15, 22, 17)

    studytime = st.sidebar.slider("Weekly Study Time (1–4)", 1, 4, 2)
    failures = st.sidebar.slider("Past Failures", 0, 3, 0)

    activities = st.sidebar.selectbox("Extracurricular Activities", ["yes", "no"])
    higher = st.sidebar.selectbox("Wants Higher Education", ["yes", "no"])

    goout = st.sidebar.slider("Going Out with Friends (1–5)", 1, 5, 3)
    Dalc = st.sidebar.slider("Workday Alcohol Consumption (1–5)", 1, 5, 1)
    health = st.sidebar.slider("Health Status (1–5)", 1, 5, 3)

    absences = st.sidebar.slider("School Absences", 0, 93, 4)
    G3 = st.sidebar.slider("Final Grade", 0, 20, 12)

    data = {
        "school": school,
        "sex": sex,
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "activities": activities,
        "higher": higher,
        "goout": goout,
        "Dalc": Dalc,
        "health": health,
        "absences": absences,
        "G3": G3
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------- ENCODING ----------------
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict Walc"):
    prediction = model.predict(input_encoded)[0]
    confidence = model.predict_proba(input_encoded).max()

    st.subheader("Prediction 🎯")
    st.success(f"Predicted Weekend Alcohol Consumption (Walc): {prediction}")
    st.info(f"Prediction Confidence: {confidence:.2f}")
