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

# ===== Custom CSS Styling for Professional Aesthetic =====
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    .stApp {
        background: linear-gradient(145deg, #1c1f26, #2e323f);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #00ffc6;
        margin-bottom: 20px;
        text-shadow: 0 0 8px #00ffc6;
    }

    .section-title {
        color: #ffffff;
        font-weight: 600;
        margin-top: 20px;
        font-size: 18px;
    }

    .predict-button {
        background-color: #00ffc6;
        color: black;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }

    .predict-button:hover {
        background-color: #00e6b0;
        box-shadow: 0 0 15px #00ffc6;
    }

    .result-box {
        background-color: #2e323f;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 0 10px rgba(0,255,198,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== App Title =====
st.markdown('<h1 class="app-title">💰 Smart Loan Recovery Prediction 🏦</h1>', unsafe_allow_html=True)


# ===== Input Fields =====
st.markdown('<div class="section-title">Borrower Information:</div>', unsafe_allow_html=True)

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

st.markdown("<hr>", unsafe_allow_html=True)

# ===== Prediction Button =====
predict_button = st.button("🚀 Predict Loan Recovery Status", key="predict")
if predict_button:
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

      st.markdown(f"""
        <div class="result-box">
            <h3>✅ Prediction Result:</h3>
            <p><strong>Borrower ID:</strong> {borrower_id if borrower_id else "N/A"}</p>
            <p><strong>Predicted Loan Recovery Status:</strong> <span style='color:#00ffc6;'>{predicted_status}</span></p>
        </div>
        """, unsafe_allow_html=True)
  except Exception as e:
        st.error(f"❌ Error during prediction: {e}")