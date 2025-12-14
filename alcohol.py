import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Weekend Alcohol Predictor",
    page_icon="🍷",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("student_alcohol_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("student_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------- CUSTOM CREATIVE THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff1f2, #f3e8ff);
    color: #2c2c54;
}

h1 {
    color: #ff6159;
    font-weight: 800;
}

h3 {
    color: #4b0082;
}

/* Center card */
.card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.08);
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #ff6159, #ff8c61);
    color: white;
    font-size: 16px;
    font-weight: 600;
    border-radius: 14px;
    padding: 12px 26px;
    border: none;
}

/* Metric box */
.result-box {
    padding: 24px;
    border-radius: 16px;
    color: white;
    font-size: 18px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🍷 Student Weekend Alcohol Consumption Predictor</h1>
    <p style='text-align:center;font-size:17px;'>Predict student weekend alcohol risk using lifestyle & academic factors</p>
    <br>
    """,
    unsafe_allow_html=True
)

# ---------------- CENTER INPUT CARD ----------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Student Information")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)

    studytime = st.slider("Weekly Study Time (1–4)", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 3, 0)

    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])

    goout = st.slider("Going Out with Friends (1–5)", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol Consumption (1–5)", 1, 5, 1)
    health = st.slider("Health Status (1–5)", 1, 5, 3)

    absences = st.slider("School Absences", 0, 93, 4)
    G3 = st.slider("Final Grade", 0, 20, 12)

    predict_btn = st.button("🔮 Predict Weekend Alcohol Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATAFRAME & ENCODING ----------------
input_df = pd.DataFrame({
    "school": [school],
    "sex": [sex],
    "age": [age],
    "studytime": [studytime],
    "failures": [failures],
    "activities": [activities],
    "higher": [higher],
    "goout": [goout],
    "Dalc": [Dalc],
    "health": [health],
    "absences": [absences],
    "G3": [G3]
})

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------- PREDICTION RESULT ----------------
if predict_btn:
    prediction = model.predict(input_encoded)[0]
    confidence = model.predict_proba(input_encoded).max()

    if prediction <= 2:
        color = "#2ecc71"
        level = "LOW"
        meaning = "Minimal risk of weekend alcohol consumption."
    elif prediction == 3:
        color = "#f1c40f"
        level = "MODERATE"
        meaning = "Moderate alcohol consumption risk."
    else:
        color = "#e74c3c"
        level = "HIGH"
        meaning = "High risk of excessive weekend alcohol consumption."

    st.markdown("## 🎯 Prediction Result")
    st.markdown(
        f"""
        <div class='result-box' style='background:{color};'>
            <b>Risk Level:</b> {level}<br>
            <b>Predicted Walc Score:</b> {prediction}<br><br>
            <b>Interpretation:</b> {meaning}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style='margin-top:12px;background:#fff3f0;padding:14px;border-radius:14px;'>
            <b>Prediction Confidence:</b> {confidence:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center;font-size:14px;'>✨ ML-powered Student Risk Analysis | Streamlit App</p>
    """,
    unsafe_allow_html=True
)
