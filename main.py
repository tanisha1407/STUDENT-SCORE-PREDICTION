import streamlit as st
import pandas as pd
import joblib


model = joblib.load("final_linear_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")


st.set_page_config(page_title="Exam Score Predictor", layout="wide")

st.title("🎓 Student Exam Score Predictor")

# Input fields
with st.form("prediction_form"):
    st.subheader("Enter Student Information")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.number_input("Hours Studied", min_value=0.0)
        attendance = st.number_input("Attendance Rate (%)", min_value=0, max_value=100)
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
        previous_scores = st.number_input("Previous Scores", min_value=0.0)

    with col2:
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0)
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0)

    col3, col4 = st.columns(2)

    with col3:
        learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
        parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    with col4:
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    submit = st.form_submit_button("Predict Score")

# On submit
if submit:
    input_data = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Gender": gender
    }])

    # One-hot encode user input
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    st.success(f"✅ Predicted Exam Score: **{prediction:.2f}**")
