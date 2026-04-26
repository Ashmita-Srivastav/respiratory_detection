
# streamlit_app.py - Run: streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import os
import tempfile
from respiratory_disease_predictor import RespiratoryDiseasePredictor

st.set_page_config(
    page_title="Respiratory Disease Predictor",
    page_icon="🫁",
    layout="wide"
)

# Header
st.title("🫁 Respiratory Disease Predictor")
st.markdown("""
**AI-powered analysis of voice/cough/breathing audio samples**
*Based on Coswara Dataset | CNN + BiLSTM Model*
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This tool analyzes audio recordings of:
    - Cough sounds
    - Breathing patterns
    - Voice samples

    to predict respiratory conditions including:
    COVID-19, Asthma, Bronchitis,
    Pneumonia, URTI, and COPD.
    """)
    st.warning("⚠️ Not a substitute for medical diagnosis!")

# Load predictor
@st.cache_resource
def load_predictor():
    return RespiratoryDiseasePredictor()

predictor = load_predictor()

# File uploader
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📁 Upload Audio Sample")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Upload a voice/cough/breathing recording"
    )

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        if st.button("🔬 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                # Save temp
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = predictor.predict(tmp_path)
                os.unlink(tmp_path)

                # Store in session state
                st.session_state['result'] = result

with col2:
    if 'result' in st.session_state:
        result = st.session_state['result']
        st.subheader("🏥 Prediction Results")

        # Primary diagnosis
        disease = result['primary_diagnosis']
        confidence = result['confidence']
        color = "red" if confidence > 0.8 else "orange" if confidence > 0.5 else "gray"
        st.markdown(f"### Diagnosis: :{color}[{disease}]")
        st.metric("Confidence", result['confidence_pct'])

        # Top predictions bar chart
        import pandas as pd
        top3 = result['top_3_predictions']
        df = pd.DataFrame(top3)
        df['confidence_pct_val'] = df['confidence'] * 100
        st.bar_chart(df.set_index('disease')['confidence_pct_val'])

        # Symptoms
        st.subheader("🩺 Likely Symptoms")
        for sym in result['inferred_symptoms']:
            st.markdown(f"- {sym.replace('_', ' ').title()}")

        # Recommendation
        st.subheader("💊 Recommendation")
        st.info(result['recommendation'])

        st.caption("*Disclaimer: This is an AI tool for screening only. Always consult a qualified medical professional.*")
