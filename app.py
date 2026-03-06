import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-header {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
        padding-left: 10px;
    }

    .result-box-dropout {
        background: linear-gradient(135deg, #ff6b6b, #ee0979);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(238, 9, 121, 0.3);
        animation: pulse 2s infinite;
    }

    .result-box-safe {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .result-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(238, 9, 121, 0.3); }
        50% { box-shadow: 0 10px 40px rgba(238, 9, 121, 0.6); }
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    .sidebar-info {
        background: #eef2ff;
        border-radius: 10px;
        padding: 12px;
        font-size: 0.85rem;
        color: #4338ca;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model, Scaler, Features
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("Logistic_reg_Dropout.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Scalling_Dropout.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🎓 Student Dropout Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered risk analysis using Logistic Regression · Fill in all student details for accurate prediction</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Input Form Layout
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# ── Column 1: Personal & Background ──────────
with col1:
    st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)

    age = st.number_input(
        "Age", min_value=15, max_value=40, value=20, step=1,
        help="Student's current age in years"
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"],
        help="Student's gender"
    )

    family_income = st.number_input(
        "Family Annual Income (₹ / currency units)", min_value=0, max_value=500000, value=40000, step=1000,
        help="Total annual family income"
    )

    parental_education = st.selectbox(
        "Parental Education Level",
        ["None", "High School", "Bachelor", "Master", "PhD"],
        index=2,
        help="Highest education level of either parent"
    )

    internet_access = st.selectbox(
        "Internet Access at Home",
        ["Yes", "No"],
        help="Does the student have reliable internet access?"
    )

    st.markdown('<p class="section-header">🏫 Academic & Course Info</p>', unsafe_allow_html=True)

    semester = st.selectbox(
        "Current Year / Semester",
        ["Year 1", "Year 2", "Year 3", "Year 4"],
        help="Which academic year is the student in?"
    )

    department = st.selectbox(
        "Department",
        ["Arts", "Business", "CS", "Engineering", "Science"],
        index=2,
        help="Student's academic department"
    )

# ── Column 2: Academic Performance & Lifestyle ──
with col2:
    st.markdown('<p class="section-header">📊 Academic Performance</p>', unsafe_allow_html=True)

    gpa = st.slider(
        "Current GPA (0.0 – 4.0)", min_value=0.0, max_value=4.0, value=2.5, step=0.01,
        help="Current semester GPA on a 4.0 scale"
    )

    semester_gpa = st.slider(
        "Semester GPA (0.0 – 4.0)", min_value=0.0, max_value=4.0, value=2.5, step=0.01,
        help="GPA for the most recent semester"
    )

    cgpa = st.slider(
        "CGPA / Cumulative GPA (0.0 – 4.0)", min_value=0.0, max_value=4.0, value=2.5, step=0.01,
        help="Cumulative GPA across all semesters"
    )

    attendance_rate = st.slider(
        "Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1,
        help="Percentage of classes attended"
    )

    assignment_delay = st.number_input(
        "Assignment Delay Days (avg.)", min_value=0, max_value=15, value=2, step=1,
        help="Average number of days assignments are submitted late"
    )

    st.markdown('<p class="section-header">⏱️ Study & Lifestyle</p>', unsafe_allow_html=True)

    study_hours = st.slider(
        "Study Hours per Day", min_value=0.0, max_value=12.0, value=4.0, step=0.1,
        help="Average number of hours spent studying per day"
    )

    travel_time = st.slider(
        "Travel Time to Campus (minutes)", min_value=5, max_value=120, value=30, step=1,
        help="Daily one-way commute time to campus"
    )

    stress_index = st.slider(
        "Stress Index (1 – 10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1,
        help="Self-reported stress level (1 = very low, 10 = extremely high)"
    )

    part_time_job = st.selectbox(
        "Part-Time Job",
        ["No", "Yes"],
        help="Does the student work part-time while studying?"
    )

    scholarship = st.selectbox(
        "Scholarship",
        ["No", "Yes"],
        help="Is the student on a scholarship?"
    )

st.markdown("---")

# ─────────────────────────────────────────────
# Encode inputs matching training preprocessing
# ─────────────────────────────────────────────
def encode_inputs():
    gender_enc    = 1 if gender == "Male" else 0
    internet_enc  = 1 if internet_access == "Yes" else 0
    job_enc       = 1 if part_time_job == "Yes" else 0
    scholarship_enc = 1 if scholarship == "Yes" else 0

    semester_map = {"Year 1": 1, "Year 2": 2, "Year 3": 3, "Year 4": 4}
    semester_enc = semester_map[semester]

    parent_map = {"None": 0, "High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    parental_enc = parent_map[parental_education]

    # One-hot encoded department columns (Arts and Business are base — not in features)
    dept_business    = 1 if department == "Business" else 0
    dept_cs          = 1 if department == "CS" else 0
    dept_engineering = 1 if department == "Engineering" else 0
    dept_science     = 1 if department == "Science" else 0

    # Build dict matching exact feature_names order
    data = {
        "Age":                    age,
        "Gender":                 gender_enc,
        "Family_Income":          family_income,
        "Internet_Access":        internet_enc,
        "Study_Hours_per_Day":    study_hours,
        "Attendance_Rate":        attendance_rate,
        "Assignment_Delay_Days":  assignment_delay,
        "Travel_Time_Minutes":    travel_time,
        "Part_Time_Job":          job_enc,
        "Scholarship":            scholarship_enc,
        "Stress_Index":           stress_index,
        "GPA":                    gpa,
        "Semester_GPA":           semester_gpa,
        "CGPA":                   cgpa,
        "Semester":               semester_enc,
        "Parental_Education":     parental_enc,
        "Department_Business":    dept_business,
        "Department_CS":          dept_cs,
        "Department_Engineering": dept_engineering,
        "Department_Science":     dept_science,
    }

    # Create DataFrame with exact column order from features.pkl
    df = pd.DataFrame([data], columns=feature_names)
    return df

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_clicked = st.button("🔍 Predict Dropout Risk")

if predict_clicked:
    input_df     = encode_inputs()
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    proba        = model.predict_proba(scaled_input)[0]
    dropout_prob = proba[1]
    safe_prob    = proba[0]

    st.markdown("### 📋 Prediction Result")
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-dropout">
                <div class="result-title">⚠️ High Dropout Risk</div>
                <div class="result-subtitle">This student is likely to drop out.<br>Immediate intervention is recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-safe">
                <div class="result-title">✅ Low Dropout Risk</div>
                <div class="result-subtitle">This student is unlikely to drop out.<br>Keep up the good work!</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#ee0979' if dropout_prob > 0.5 else '#11998e'};">
                {dropout_prob*100:.1f}%
            </div>
            <div class="metric-label">Dropout Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #11998e;">
                {safe_prob*100:.1f}%
            </div>
            <div class="metric-label">Retention Probability</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk gauge bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Risk Level Gauge**")
    bar_color = "#ee0979" if dropout_prob > 0.5 else "#11998e"
    st.markdown(f"""
    <div style="background:#e5e7eb; border-radius:10px; height:20px; overflow:hidden; margin-bottom:8px;">
        <div style="width:{dropout_prob*100:.1f}%; background:{bar_color}; height:100%; border-radius:10px;
                    transition: width 0.5s ease; display:flex; align-items:center; justify-content:flex-end; 
                    padding-right:8px; color:white; font-size:0.75rem; font-weight:600;">
            {dropout_prob*100:.1f}%
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#6b7280;">
        <span>0% – No Risk</span>
        <span>50% – Moderate</span>
        <span>100% – High Risk</span>
    </div>
    """, unsafe_allow_html=True)

    # Key risk factors insight
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🔎 Key Input Summary**")
    summary_data = {
        "Feature": ["Attendance Rate", "CGPA", "Study Hours/Day", "Stress Index", "Assignment Delay"],
        "Value":   [f"{attendance_rate:.1f}%", f"{cgpa:.2f}", f"{study_hours:.1f} hrs",
                    f"{stress_index:.1f}/10", f"{assignment_delay} days"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "🎓 Student Dropout Prediction &nbsp;·&nbsp; Model: Logistic Regression &nbsp;·&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)