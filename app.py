import streamlit as st
import pickle
import numpy as np

# ===== Load the Saved Model, Scaler, and Label Encoders =====
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# ===== Streamlit App UI =====
st.title("📊 Smart Loan Recovery Prediction App")

st.write("Enter Borrower Details Below:")

# (Optional) Borrower ID just for display purpose
borrower_id = st.text_input("Borrower ID (Optional - for your reference)")

# ===== User Inputs =====
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
monthly_income = st.number_input("Monthly Income", min_value=0, value=30000)
num_dependents = st.number_input("Number of Dependents", min_value=0, value=0)

loan_amount = st.number_input("Loan Amount", min_value=1000, value=50000)
loan_tenure = st.number_input("Loan Tenure (Months)", min_value=1, value=24)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1, value=10.0)
loan_type = st.selectbox("Loan Type", ['Home', 'Auto', 'Personal', 'Business'])

collateral_value = st.number_input("Collateral Value", min_value=0, value=20000)
outstanding_loan = st.number_input("Outstanding Loan Amount", min_value=0, value=25000)
monthly_emi = st.number_input("Monthly EMI", min_value=0, value=2500)
num_missed = st.number_input("Number of Missed Payments", min_value=0, value=0)
days_past_due = st.number_input("Days Past Due", min_value=0, value=0)
collection_attempts = st.number_input("Number of Collection Attempts", min_value=0, value=0)
collection_method = st.selectbox("Collection Method", ['Settlement Offer', 'Legal Notice', 'Calls', 'Debt Collectors'])
legal_action = st.selectbox("Legal Action Taken", ["Yes", "No"])

# ===== Prediction Button =====
if st.button("Predict Loan Recovery Status"):
  try:
        # ===== Step 1: Encode Categorical Features =====
        # Step 1: Encode Categorical Input
      gender_enc = label_encoders['Gender'].transform([gender])[0]
      employment_enc = label_encoders['Employment_Type'].transform([employment])[0]
      loan_type_enc = label_encoders['Loan_Type'].transform([loan_type])[0]
      collection_enc = label_encoders['Collection_Method'].transform([collection_method])[0]
      legal_action_enc = label_encoders['Legal_Action_Taken'].transform([legal_action])[0]

# Step 2: Combine Numeric + Encoded Categoricals (Total 17 Features - Same as Model Training)
      final_input = np.array([[age, monthly_income, num_dependents, loan_amount, loan_tenure,
                         interest_rate, collateral_value, outstanding_loan, monthly_emi,
                         num_missed, days_past_due, collection_attempts,
                         gender_enc, employment_enc, loan_type_enc, collection_enc, legal_action_enc]])

# Step 3: Scale Final Input
      final_scaled_input = scaler.transform(final_input)

# Step 4: Make Prediction
      prediction = model.predict(final_scaled_input)[0]

# Step 5: Decode Target Label
      predicted_status = label_encoders['Recovery_Status'].inverse_transform([prediction])[0]

      st.success(f"✅ Predicted Loan Recovery Status: {predicted_status}")

  except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
