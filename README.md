# 🏥 Clinic Operations & Patient No-Show Predictive Analytics

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 The Business Problem
In the healthcare industry, missed appointments (no-shows) lead to significant revenue loss, idle staff time, and reduced quality of care for other patients. Being able to predict which patients are at a high risk of missing their appointments allows clinic managers to proactively double-book slots or send targeted reminders, directly optimizing operational efficiency.

## 🚀 Project Overview
This project is an end-to-end data analytics and machine learning solution. It analyzes over 100,000 clinical appointment records to identify the key drivers of no-shows and deploys a predictive model via an interactive web application.

**(Note to self: Take a screenshot of your Streamlit app and drag-and-drop it right here to upload it!)**

## 📊 Key Insights & Exploratory Data Analysis (EDA)
During the data cleaning and EDA phase, several critical operational insights were discovered:
* **Wait Time is the #1 Predictor:** The longer the lead time between scheduling an appointment and the actual appointment date, the higher the probability of a no-show. 
* **Data Imbalance:** The baseline data showed a natural imbalance (roughly 80% show-up vs. 20% no-show), requiring careful model evaluation using Precision and Recall rather than basic Accuracy.
* **Feature Engineering:** Extracted temporal features such as `WaitTimeDays` and `AppointmentDayOfWeek` heavily improved model understanding.

## 🧠 Machine Learning Model
* **Algorithm:** Random Forest Classifier (`n_estimators=100`)
* **Features Used:** Age, Wait Time, Medical History (Hypertension, Diabetes, Alcoholism, Handicap), Welfare Scholarship Status, and SMS Reminder Status.
* **Performance:** The model successfully learned the feature importance hierarchy, prioritizing operational metrics (Wait Time) and demographic metrics (Age) over basic medical history when predicting attendance.

## 🛠️ How to Run the Dashboard Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Clinic-Operations-Predictive-Analytics.git](https://github.com/YOUR_USERNAME/Clinic-Operations-Predictive-Analytics.git)

2. Navigate to the directory:
   ```bash
   cd Clinic-Operations-Predictive-Analytics

2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn streamlit joblib

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
