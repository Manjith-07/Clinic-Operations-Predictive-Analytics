import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinic No-Show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Custom CSS — Clinical dark aesthetic with Syne + DM Mono
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background-color: #0f1117;
    color: #e8e8e8;
}

/* ── Main content area ── */
.main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161920;
    border-right: 1px solid #2a2d36;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: #8a8fa8 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #c8cad4 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 1.5rem !important;
}

/* ── Sliders ── */
.stSlider div[data-baseweb="slider"] > div > div:first-child {
    background: transparent !important;
}
.stSlider > div > div > div > div {
    background: #1D9E75 !important;
    border: 2px solid #0F6E56 !important;
}

/* ── Selectboxes ── */
.stSelectbox > div > div {
    background-color: #1e2130 !important;
    border: 1px solid #2e3245 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Checkboxes ── */
.stCheckbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: #8a8fa8 !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #1D9E75 0%, #0F6E56 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s;
}

/* Target the text wrapper specifically to override the global 'p' style */
div[data-testid="stButton"] button p {
    color: #000000 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

div[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
}

/* ── Divider ── */
hr {
    border-color: #2a2d36 !important;
    margin: 1.5rem 0 !important;
}

/* ── Metric cards override ── */
[data-testid="metric-container"] {
    background: #1a1d28;
    border: 1px solid #2a2d36;
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7080 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e8e8e8 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Alert / info / success / error boxes ── */
.stAlert {
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Dataframe / table ── */
.stDataFrame {
    border: 1px solid #2a2d36 !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background-color: #1D9E75 !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161920;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2d36;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #6b7080 !important;
    padding: 6px 16px;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2130 !important;
    color: #e8e8e8 !important;
}

/* ── Headings ── */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #e8e8e8 !important; letter-spacing: -0.5px; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #c8cad4 !important; }
h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #a0a3b0 !important; font-size: 14px !important; letter-spacing: 0.05em; text-transform: uppercase; }
p  { font-family: 'DM Mono', monospace !important; font-size: 13px !important; color: #8a8fa8 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('healthcare_rf_model.pkl')

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False


# ---------------------------------------------------------
# 4. Helper: heuristic fallback scorer (mirrors RF logic)
#    Remove this block when your real model is available.
# ---------------------------------------------------------
def heuristic_score(age, wait, sms, scholarship, gender_male,
                    hypertension, diabetes, alcoholism, handicap):
    score = 0.22
    if age < 18:   score += 0.12
    elif age < 30: score += 0.08
    elif age > 60: score -= 0.06
    if wait > 14:  score += 0.08 + (wait - 14) * 0.003
    elif wait < 3: score -= 0.04
    if sms:        score -= 0.10
    if scholarship:score += 0.06
    if gender_male:score += 0.03
    if alcoholism: score += 0.09
    if hypertension: score -= 0.03
    if diabetes:   score -= 0.02
    if handicap:   score += 0.04
    return float(np.clip(score, 0.03, 0.97))


# ---------------------------------------------------------
# 5. Session state for history log
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------
# 6. Header
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;'>
        <div style='width:8px;height:8px;border-radius:50%;background:#1D9E75;
                    box-shadow:0 0 8px #1D9E75;animation:none;'></div>
        <span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;
                     color:#e8e8e8;letter-spacing:-0.5px;'>
            Patient No-Show Predictor
        </span>
    </div>
    <div style='font-family:"DM Mono",monospace;font-size:11px;color:#4a4f63;
                letter-spacing:0.08em;margin-left:20px;'>
        ML-POWERED · REAL-TIME RISK ASSESSMENT
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    if not model_loaded:
        st.warning("⚠️ Using demo model", icon="⚠️")
    else:
        st.success("Model loaded", icon="✅")

st.divider()


# ---------------------------------------------------------
# 7. Sidebar — patient inputs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 👤 Patient Profile")

    age = st.slider("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    wait_time = st.slider("Wait Time (days since booking)", 0, 90, 7)

    st.markdown("---")
    st.markdown("## 📲 Interventions")
    sms_received = st.selectbox("SMS Reminder", ["Not sent", "Sent"])
    scholarship = st.selectbox("Welfare Programme", ["Not enrolled", "Enrolled"])

    st.markdown("---")
    st.markdown("## 🩺 Medical History")
    hypertension = st.checkbox("Hypertension")
    diabetes     = st.checkbox("Diabetes")
    alcoholism   = st.checkbox("Alcoholism")
    handicap     = st.checkbox("Handicap")

    st.markdown("---")
    predict_btn = st.button("Save Risk Assessment")


# ---------------------------------------------------------
# 8. Build input DataFrame
# ---------------------------------------------------------
gender_num   = 1 if gender == "Male" else 0
sms_num      = 1 if sms_received == "Sent" else 0
scholar_num  = 1 if scholarship == "Enrolled" else 0

input_data = pd.DataFrame({
    "Age":          [age],
    "Scholarship":  [scholar_num],
    "Hypertension": [1 if hypertension else 0],
    "Diabetes":     [1 if diabetes else 0],
    "Alcoholism":   [1 if alcoholism else 0],
    "Handicap":     [1 if handicap else 0],
    "SMSReceived":  [sms_num],
    "WaitTimeDays": [wait_time],
    "Gender_Num":   [gender_num],
})


# ---------------------------------------------------------
# 9. Compute prediction
# ---------------------------------------------------------
if model_loaded:
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    noshow_prob = probability[1]
    show_prob   = probability[0]
else:
    noshow_prob = heuristic_score(
        age, wait_time, bool(sms_num), bool(scholar_num), bool(gender_num),
        hypertension, diabetes, alcoholism, handicap
    )
    show_prob   = 1 - noshow_prob
    prediction  = 1 if noshow_prob >= 0.5 else 0

is_noshow = prediction == 1


# ---------------------------------------------------------
# 10. Save to history on button click
# ---------------------------------------------------------
if predict_btn:
    st.session_state.history.insert(0, {
        "Age": age,
        "Gender": gender,
        "Wait": wait_time,
        "SMS": sms_received,
        "No-Show %": f"{noshow_prob * 100:.1f}%",
        "Outcome": "⚠️ No-Show" if is_noshow else "✅ Attends",
    })
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]


# ---------------------------------------------------------
# 11. Tabs: Result · Factors · History
# ---------------------------------------------------------
tab_result, tab_factors, tab_history = st.tabs(
    ["📊  Result", "🔬  Risk Factors", "🕘  History"]
)


# ── TAB 1: Result ─────────────────────────────────────────
with tab_result:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("No-Show Risk", f"{noshow_prob * 100:.1f}%",
                  delta="HIGH" if is_noshow else "LOW",
                  delta_color="inverse")
    with m2:
        st.metric("Likely to Attend", f"{show_prob * 100:.1f}%")
    with m3:
        st.metric("Wait Time", f"{wait_time}d",
                  delta="Long wait" if wait_time > 14 else "Short wait",
                  delta_color="inverse" if wait_time > 14 else "normal")
    with m4:
        st.metric("Patient Age", f"{age} yrs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability gauge bar ──
    bar_color = "#D85A30" if is_noshow else "#1D9E75"
    pct       = noshow_prob * 100
    st.markdown(f"""
    <div style='margin-bottom:6px;'>
        <div style='display:flex;justify-content:space-between;
                    font-family:"DM Mono",monospace;font-size:11px;
                    color:#4a4f63;letter-spacing:0.06em;margin-bottom:8px;'>
            <span>NO-SHOW PROBABILITY</span>
            <span style='color:{bar_color};font-weight:600;font-size:14px;'>{pct:.1f}%</span>
        </div>
        <div style='background:#1a1d28;border-radius:6px;height:10px;overflow:hidden;
                    border:1px solid #2a2d36;'>
            <div style='width:{pct}%;height:100%;background:{bar_color};
                        border-radius:6px;transition:width 0.5s ease;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-family:"DM Mono",monospace;font-size:10px;color:#4a4f63;
                    margin-top:4px;'>
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Verdict card ──
    if is_noshow:
        st.error(f"""
**⚠️ HIGH RISK — Patient unlikely to attend**

Predicted no-show probability: **{noshow_prob * 100:.1f}%**

Recommended actions:
- Initiate a manual confirmation call
- Flag slot for potential overbooking
- Re-send SMS reminder with personalised message
        """)
    else:
        st.success(f"""
**✅ LOW RISK — Patient expected to attend**

Predicted attendance probability: **{show_prob * 100:.1f}%**

Recommended actions:
- Proceed with standard appointment preparation
- Standard SMS reminder 24 h before slot
- No additional intervention required
        """)


# ── TAB 2: Risk Factors ───────────────────────────────────
with tab_factors:
    st.markdown("### Factor Impact Analysis")

    factors = [
        ("Age",          age,         age < 30,        "years old",      "Younger patients miss more",  "Age > 30 reduces risk"),
        ("Wait Time",    wait_time,   wait_time > 14,  "days",           "Long wait → higher no-show", "Short wait is positive"),
        ("SMS Reminder", sms_num,     sms_num == 0,    "",               "No reminder sent",            "Reminder reduces risk by ~10pp"),
        ("Welfare Pgm",  scholar_num, scholar_num == 1,"",               "Welfare enrolled → slight ↑", "Not enrolled is neutral"),
        ("Alcoholism",   int(alcoholism), alcoholism,  "",               "Raises risk ~9pp",            "No impact"),
        ("Handicap",     int(handicap),   handicap,    "",               "Minor risk increase",         "No impact"),
        ("Hypertension", int(hypertension), False,     "",               "",                            "Slight protective effect"),
        ("Diabetes",     int(diabetes),     False,     "",               "",                            "Slight protective effect"),
    ]

    for name, val, is_risky, unit, risk_note, safe_note in factors:
        col_a, col_b, col_c = st.columns([2, 3, 3])
        with col_a:
            display = f"{val} {unit}".strip() if unit else ("Yes" if val else "No")
            color   = "#D85A30" if is_risky else "#1D9E75"
            icon    = "↑" if is_risky else "↓"
            st.markdown(f"""
            <div style='background:#1a1d28;border:1px solid #2a2d36;border-radius:8px;
                        padding:10px 14px;'>
                <div style='font-family:"DM Mono",monospace;font-size:10px;
                            color:#4a4f63;letter-spacing:0.06em;margin-bottom:4px;'>
                    {name.upper()}
                </div>
                <div style='font-family:Syne,sans-serif;font-size:15px;font-weight:700;
                            color:#e8e8e8;'>
                    {display}
                </div>
                <div style='font-family:"DM Mono",monospace;font-size:11px;
                            color:{color};margin-top:3px;'>
                    {icon} {"Higher risk" if is_risky else "Lower risk"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            if is_risky and risk_note:
                st.markdown(f"""
                <div style='background:#1a0e08;border-left:3px solid #D85A30;
                            border-radius:0 8px 8px 0;padding:10px 14px;height:100%;'>
                    <div style='font-family:"DM Mono",monospace;font-size:12px;
                                color:#D85A30;'>{risk_note}</div>
                </div>
                """, unsafe_allow_html=True)
        with col_c:
            if not is_risky and safe_note:
                st.markdown(f"""
                <div style='background:#041a0f;border-left:3px solid #1D9E75;
                            border-radius:0 8px 8px 0;padding:10px 14px;height:100%;'>
                    <div style='font-family:"DM Mono",monospace;font-size:12px;
                                color:#1D9E75;'>{safe_note}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Current Input Summary")
    st.dataframe(input_data, use_container_width=True)


# ── TAB 3: History ────────────────────────────────────────
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#4a4f63;
                    font-family:"DM Mono",monospace;font-size:13px;'>
            No assessments saved yet. Use the "Save Risk Assessment" button on the sidebar to save the predicted outcomes.
        </div>
        """, unsafe_allow_html=True)
    else:
        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # Mini stats
        st.markdown("<br>", unsafe_allow_html=True)
        total     = len(st.session_state.history)
        noshow_ct = sum(1 for r in st.session_state.history if "No-Show" in r["Outcome"])
        attend_ct = total - noshow_ct

        hc1, hc2, hc3 = st.columns(3)
        with hc1: st.metric("Total Assessed", total)
        with hc2: st.metric("High Risk", noshow_ct)
        with hc3: st.metric("Low Risk",  attend_ct)


# ---------------------------------------------------------
# 12. Footer
# ---------------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center;font-family:"DM Mono",monospace;font-size:10px;
            color:#2e3245;letter-spacing:0.08em;'>
    CLINIC OPERATIONS · PREDICTIVE ANALYTICS · FOR INTERNAL USE ONLY
</div>
""", unsafe_allow_html=True)