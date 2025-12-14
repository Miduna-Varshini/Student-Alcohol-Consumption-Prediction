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
    background: linear-gradient(135deg, #ffd1dc, #e6ccff);
    color: #2c2c54;
}

/* Title */
h1 {
    color: #4b0082;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #2c2c54;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Button */
div.stButton > button {
    background-color: #ff69b4;
    color: white;
    font-size: 16px;
    border-radius: 12px;
    padding: 10px 24px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🍷 Student Weekend Alcohol Consumption Predictor")
st.markdown(
    "Predict **weekend alcohol consumption (Walc)** using key student lifestyle and academic factors."
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

# ---------------- PREDICTION & OUTPUT ----------------
if st.button("Predict Walc"):
    prediction = model.predict(input_encoded)[0]
    confidence = model.predict_proba(input_encoded).max()

    # Risk level mapping
    if prediction == 1:
        color = "#2ecc71"
        level = "VERY LOW"
        meaning = "Student shows minimal risk of weekend alcohol consumption."
    elif prediction == 2:
        color = "#7bed9f"
        level = "LOW"
        meaning = "Student has a low level of alcohol consumption risk."
    elif prediction == 3:
        color = "#f1c40f"
        level = "MODERATE"
        meaning = "Student shows a moderate risk and should be monitored."
    elif prediction == 4:
        color = "#e67e22"
        level = "HIGH"
        meaning = "Student shows high weekend alcohol consumption behavior."
    else:
        color = "#e74c3c"
        level = "VERY HIGH"
        meaning = "Student is at high risk of excessive weekend alcohol consumption."

    st.markdown("## 🎯 Prediction Result")

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:22px;
            border-radius:14px;
            color:white;
            font-size:18px;
        ">
            <b>Risk Level:</b> {level}<br>
            <b>Walc Score:</b> {prediction}<br><br>
            <b>Meaning:</b> {meaning}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            margin-top:15px;
            background-color:#f3e8ff;
            padding:14px;
            border-radius:12px;
            color:#4b0082;
            font-size:16px;
        ">
            <b>Prediction Confidence:</b> {confidence:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
