
import streamlit as st
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime

# Page configuration must come first
st.set_page_config(
    page_title="IntelliCare AI",
    page_icon="🩺",
    layout="wide"
)

# Load custom CSS styling
css_path = Path(__file__).parent / "assets" / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )
else:
    st.warning("CSS file not found. Please check assets/style.css.")

# Load the trained educational ML model
MODEL_PATH = Path(__file__).parent / "symptom_model.pkl"

if not MODEL_PATH.exists():
    st.error("ML model not found. Please check symptom_model.pkl.")
    st.stop()

with open(MODEL_PATH, "rb") as model_file:
    ml_model = pickle.load(model_file)

# Main welcome section
st.title("🩺 IntelliCare AI")
st.subheader("💚 Your Smart Healthcare Assistant")

st.write(
    "Explore AI-powered symptom patterns and manage "
    "healthcare demo records in one place."
)

st.info(
    "🤖 Artificial Intelligence | 📊 Machine Learning | 💚 Healthcare"
)

st.caption(
    "Educational prototype only. This application does not "
    "diagnose medical conditions."
)

# Create the SQLite database and demo patient table
conn = sqlite3.connect("intellicare.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS patient_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    age INTEGER,
    symptoms TEXT NOT NULL,
    created_at TEXT NOT NULL
)
""")
conn.commit()

# Patient entry form
st.header("📝 Demo Patient Entry")

with st.form("patient_form"):
    patient_name = st.text_input("Demo patient name")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=18
    )

    symptoms = st.multiselect(
        "Select symptoms for the educational demo",
        [
            "Fever",
            "Cough",
            "Headache",
            "Fatigue",
            "Sore throat",
            "Nausea",
            "Body ache"
        ]
    )

    submitted = st.form_submit_button("Save Demo Record")

if submitted:
    if not patient_name.strip() or not symptoms:
        st.warning("Enter a demo name and select at least one symptom.")
    else:
        conn.execute(
            """
            INSERT INTO patient_records
            (patient_name, age, symptoms, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                patient_name.strip(),
                int(age),
                ", ".join(symptoms),
                datetime.now().isoformat(timespec="seconds")
            )
        )
        conn.commit()
        st.success("Demo record saved successfully!")

# AI symptom pattern classifier
st.header("🤖 AI Symptom Pattern Classifier")

st.write(
    "Select symptoms to test the trained educational ML model. "
    "This is a demonstration, not a medical diagnosis."
)

ml_symptoms = st.multiselect(
    "Choose symptoms for ML analysis",
    [
        "fever",
        "cough",
        "headache",
        "fatigue",
        "sore throat",
        "nausea",
        "body ache",
        "sneezing",
        "runny nose",
        "throat irritation"
    ],
    key="ml_symptoms"
)

if st.button("Analyze Symptom Pattern"):
    if not ml_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        symptom_text = " ".join(ml_symptoms)
        prediction = ml_model.predict([symptom_text])[0]

        st.success(f"Demo model output: {prediction}")

        st.caption(
            "Educational prototype only. This result is not a diagnosis "
            "and must not guide treatment decisions."
        )

# Demo records dashboard
st.header("📊 Demo Records Dashboard")

records = pd.read_sql_query(
    "SELECT * FROM patient_records ORDER BY id DESC",
    conn
)

col1, col2 = st.columns(2)

col1.metric("Total demo records", len(records))

if not records.empty:
    all_symptoms = (
        records["symptoms"]
        .str.split(", ")
        .explode()
    )

    col2.metric(
        "Different symptoms recorded",
        all_symptoms.nunique()
    )

    st.dataframe(records, use_container_width=True)

    st.subheader("Most frequently selected symptoms")

    symptom_counts = all_symptoms.value_counts()
    st.bar_chart(symptom_counts)

else:
    col2.metric("Different symptoms recorded", 0)
    st.info(
        "No demo records yet. Add a record above to see the dashboard."
    )

conn.close()

st.divider()

st.caption(
    "IntelliCare AI | Educational prototype only — not a medical "
    "diagnosis or a substitute for advice from a qualified healthcare "
    "professional."
)
