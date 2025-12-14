import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("student_alcohol_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("student_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Page config
st.set_page_config(page_title="Student Weekend Alcohol Predictor", page_icon="🍷", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* Background & main container */
    .stApp {
        background: linear-gradient(to bottom right, #ffc0cb, #ffffff);
        color: #00008b;
    }
    /* Title */
    .css-18e3th9 {
        color: #ff1493;
        font-size: 3rem;
        font-weight: bold;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #e6f0ff;
        padding: 20px;
        border-radius: 15px;
    }
    /* Button style */
    div.stButton > button {
        background-color: #ff69b4;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🍷 Student Weekend Alcohol Consumption Predictor")
st.markdown("""
Predict a student's weekend alcohol consumption level (1 - very low, 5 - very high) using personal and academic details.
""")

# Sidebar
st.sidebar.header("Enter Student Information")

def user_input_features():
    school = st.sidebar.selectbox('School', ['GP', 'MS'])
    sex = st.sidebar.selectbox('Sex', ['F', 'M'])
    age = st.sidebar.slider('Age', 15, 22, 16)
    address = st.sidebar.selectbox('Address', ['U', 'R'])
    famsize = st.sidebar.selectbox('Family Size', ['LE3', 'GT3'])
    Pstatus = st.sidebar.selectbox('Parent Cohabitation Status', ['T', 'A'])
    Medu = st.sidebar.selectbox("Mother's Education", [0,1,2,3,4])
    Fedu = st.sidebar.selectbox("Father's Education", [0,1,2,3,4])
    
    # Job dropdowns with nicer names
    Mjob = st.sidebar.selectbox('Mother Job', ['teacher','health','services','at_home','other'])
    Fjob = st.sidebar.selectbox('Father Job', ['teacher','health','services','at_home','other'])
    
    reason = st.sidebar.selectbox('Reason to choose school', ['home','reputation','course','other'])
    guardian = st.sidebar.selectbox('Guardian', ['mother','father','other'])
    
    traveltime = st.sidebar.slider('Travel Time (1-4)', 1, 4, 2)
    studytime = st.sidebar.slider('Weekly Study Time (1-4)', 1, 4, 2)
    failures = st.sidebar.slider('Past Failures', 0, 3, 0)
    
    # Support & activities as dropdown
    schoolsup = st.sidebar.selectbox('School Support', ['yes','no'])
    famsup = st.sidebar.selectbox('Family Support', ['yes','no'])
    paid = st.sidebar.selectbox('Extra Paid Classes', ['yes','no'])
    activities = st.sidebar.selectbox('Extracurricular Activities', ['yes','no'])
    nursery = st.sidebar.selectbox('Attended Nursery', ['yes','no'])
    higher = st.sidebar.selectbox('Wants Higher Education', ['yes','no'])
    internet = st.sidebar.selectbox('Internet Access at Home', ['yes','no'])
    romantic = st.sidebar.selectbox('In Romantic Relationship', ['yes','no'])
    
    famrel = st.sidebar.slider('Family Relationship (1-5)', 1, 5, 4)
    freetime = st.sidebar.slider('Free Time (1-5)', 1, 5, 3)
    goout = st.sidebar.slider('Going Out (1-5)', 1, 5, 3)
    Dalc = st.sidebar.slider('Workday Alcohol Consumption (1-5)', 1, 5, 1)
    health = st.sidebar.slider('Current Health (1-5)', 1, 5, 3)
    absences = st.sidebar.slider('School Absences', 0, 93, 4)
    G1 = st.sidebar.slider('1st Period Grade', 0, 20, 11)
    G2 = st.sidebar.slider('2nd Period Grade', 0, 20, 11)
    G3 = st.sidebar.slider('Final Grade', 0, 20, 11)

    data = {
        'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': traveltime, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': famrel, 'freetime': freetime, 'goout': goout,
        'Dalc': Dalc, 'health': health, 'absences': absences, 'G1': G1, 'G2': G2, 'G3': G3
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encode input
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction button
if st.button("Predict Walc"):
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded).max()
    st.subheader("Prediction 🎯")
    st.success(f"Predicted Weekend Alcohol Consumption (Walc): {prediction}")
    st.info(f"Prediction Confidence: {prediction_proba:.2f}")
