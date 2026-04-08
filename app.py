import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Clinic No-Show Predictor", layout="wide")

# 2. Load the trained machine learning model
@st.cache_resource # This makes the app run faster by only loading the model once
def load_model():
    return joblib.load('healthcare_rf_model.pkl')

model = load_model()

# 3. Build the User Interface
st.title("🏥 Patient No-Show Predictive Analytics")
st.markdown("Enter patient details in the sidebar to predict the likelihood of an appointment no-show.")

# 4. Sidebar Inputs (The Interactive Part)
st.sidebar.header("Patient Profile")

# Creating inputs for all the features our model learned from
age = st.sidebar.slider("Age", 0, 100, 30)
wait_time = st.sidebar.slider("Wait Time (Days)", 0, 90, 7)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
scholarship = st.sidebar.selectbox("Enrolled in Welfare Program?", ["No", "Yes"])
sms_received = st.sidebar.selectbox("SMS Reminder Received?", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.header("Medical History")
hypertension = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
alcoholism = st.sidebar.checkbox("Alcoholism")
handicap = st.sidebar.checkbox("Handicap")

# 5. Data Preprocessing (Translating UI into Numbers for the Model)
# We must map the user's choices back to 1s and 0s exactly how we trained the model
input_data = pd.DataFrame({
    'Age': [age],
    'Scholarship': [1 if scholarship == "Yes" else 0],
    'Hypertension': [1 if hypertension else 0],
    'Diabetes': [1 if diabetes else 0],
    'Alcoholism': [1 if alcoholism else 0],
    'Handicap': [1 if handicap else 0],
    'SMSReceived': [1 if sms_received == "Yes" else 0],
    'WaitTimeDays': [wait_time],
    'Gender_Num': [1 if gender == "Male" else 0]
})

# 6. Make the Prediction
if st.button("Predict Appointment Outcome"):
    # The model outputs a 0 (Showed Up) or 1 (No-Show)
    prediction = model.predict(input_data)[0]
    
    # We can also get the probability (e.g., "75% sure they will show up")
    probability = model.predict_proba(input_data)[0]
    
    st.markdown("### Prediction Results:")
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of No-Show! (Probability: {probability[1] * 100:.1f}%)")
        st.write("**Recommendation:** Contact the patient for manual confirmation or double-book this time slot.")
    else:
        st.success(f"✅ Likely to Show Up (Probability: {probability[0] * 100:.1f}%)")