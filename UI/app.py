import streamlit as st
import requests

BASE = "http://localhost:8000"

st.title("Smart Unified Diagnosis Support System (SUDSS) 🩺")

page = st.sidebar.radio("Navigate", ["Upload History", "Generate Report", "Review Reports"])

if page == "Upload History":
    st.header("Upload Patient History")
    st.info("For local testing, drop FHIR JSON into /data/patients and run ingestion manually.")

elif page == "Generate Report":
    st.header("Generate Diagnostic Report")
    draft = st.text_area("Draft Report")
    query = st.text_input("Query / Symptoms")
    if st.button("Refine"):
        resp = requests.post(f"{BASE}/refine", json={"draft_report": draft, "query": query})
        st.json(resp.json())

elif page == "Review Reports":
    st.header("Review Reports & Apply Feedback")
    report = st.text_area("Refined Report")
    corrections = st.text_area("Corrections")
    if st.button("Submit Feedback"):
        resp = requests.post(f"{BASE}/feedback", json={"report": report, "corrections": corrections})
        st.json(resp.json())
