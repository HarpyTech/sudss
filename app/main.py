import streamlit as st
from langchain.agents import initialize_agent
from langchain.llms import VertexAI
from app.agents.extractor_agent import extract_content
from app.utils.file_utils import detect_file_type
from app.tools.biogpt_tool import biogpt_tool
from app.tools.pubmedgpt_tool import pubmedgpt_tool
from app.tools.summarizer_tool import summarizer_tool

st.set_page_config(page_title="Clinical Diagnosis Support", layout="centered")
st.title("🩺 Clinical Diagnosis Support System")

uploaded_file = st.file_uploader("Upload patient data (Text, Image, or Lab Report PDF)", type=["txt", "pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = detect_file_type(uploaded_file.name)

    with st.spinner("🔍 Extracting patient data..."):
        content = extract_content(file_bytes, file_type)

    tools = [biogpt_tool, pubmedgpt_tool, summarizer_tool]
    llm = VertexAI(model_name="gemini-pro")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="chat-zero-shot-react-description",
        verbose=True
    )

    with st.spinner("🤖 Generating diagnosis summary..."):
        result = agent.run(f"Analyze the following patient data and generate diagnostic summary:\n{content}")

    st.success("✅ Diagnostic Summary Ready")
    st.markdown("### 📝 Summary Report")
    st.text_area("Summary", value=result, height=300)
    st.download_button("Download Summary", data=result, file_name="diagnosis_summary.txt")
