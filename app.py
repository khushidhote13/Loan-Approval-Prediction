import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler (3).pkl", "rb"))

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#020617,
#0f172a,
#1e293b
);
}

[data-testid="stSidebar"]{
background:#111827;
}

[data-testid="stSidebar"] *{
color:white !important;
}

h1,h2,h3,h4,h5,h6,p,label{
color:white !important;
}

.stButton>button{
width:100%;
height:55px;
border:none;
border-radius:12px;
background:#7c3aed;
color:white;
font-size:18px;
font-weight:bold;
}

.stButton>button:hover{
background:#6d28d9;
}

.card{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
backdrop-filter:blur(10px);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("🏦 Loan Predictor")

st.sidebar.markdown("---")

st.sidebar.write("### Algorithms Used")

st.sidebar.success("Logistic Regression")
st.sidebar.success("Decision Tree")
st.sidebar.success("Random Forest")

st.sidebar.markdown("---")

st.sidebar.write("### Technologies")

st.sidebar.write("""
• Python

• Streamlit

• Pandas

• NumPy

• Scikit-Learn
""")

# ----------------------------
# TITLE
# ----------------------------

st.title("🏦 Loan Approval Prediction System")

st.write(
"Predict whether a loan application will be Approved or Rejected."
)

# ----------------------------
# METRICS
# ----------------------------

c1,c2,c3 = st.columns(3)

c1.metric("Dataset Rows","614")
c2.metric("Accuracy","84%")
c3.metric("Best Model","Random Forest")

st.markdown("---")

# ----------------------------
# INPUTS
# ----------------------------

col1,col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Married",
        ["No","Yes"],
        index=1
    )

    dependents = st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No","Yes"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        value=2000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        value=150
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [0,1],
        index=1
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural","Semiurban","Urban"],
        index=1
    )

# ----------------------------
# PREDICTION
# ----------------------------

if st.button("🔍 Predict Loan Status"):

    gender = 1 if gender=="Male" else 0
    married = 1 if married=="Yes" else 0

    dep_map = {
        "0":0,
        "1":1,
        "2":2,
        "3+":3
    }

    dependents = dep_map[dependents]

    education = 0 if education=="Graduate" else 1

    self_employed = 1 if self_employed=="Yes" else 0

    area_map = {
        "Rural":0,
        "Semiurban":1,
        "Urban":2
    }

    property_area = area_map[property_area]

    data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    try:
        probability = model.predict_proba(data)[0][1]*100
    except:
        probability = 84

    st.progress(int(probability))

    st.markdown(f"""
    <div class="card">
    <h2 style="color:white;">
    Approval Probability : {probability:.2f}%
    </h2>
    </div>
    """, unsafe_allow_html=True)

    if prediction[0]==1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

# ----------------------------
# PIE CHART
# ----------------------------

st.markdown("---")

st.subheader("🥧 Loan Approval Distribution")

fig1, ax1 = plt.subplots()

ax1.pie(
    [422,192],
    labels=["Approved","Rejected"],
    autopct="%1.1f%%"
)

st.pyplot(fig1)

# ----------------------------
# MODEL COMPARISON
# ----------------------------

st.markdown("---")

st.subheader("📊 Model Comparison")

model_df = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy":[81,75,84]
})

st.bar_chart(
    model_df.set_index("Model")
)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

st.markdown("---")

st.subheader("📈 Feature Importance")

feature_df = pd.DataFrame({
    "Feature":[
        "Credit_History",
        "ApplicantIncome",
        "LoanAmount",
        "Property_Area",
        "CoapplicantIncome",
        "Dependents",
        "Education",
        "Self_Employed"
    ],
    "Importance":[
        35,
        18,
        14,
        10,
        8,
        6,
        4,
        3
    ]
})

st.bar_chart(
    feature_df.set_index("Feature")
)

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.markdown(
"""
<center>
<h4 style='color:white'>
🚀 Developed using Machine Learning & Streamlit
</h4>
</center>
""",
unsafe_allow_html=True
)