import streamlit as st
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.extractor_agent import extract_content
from utils.file_utils import detect_file_type
from tools import summarizer_tool, pubmed_a_gemma_tool
from config import variables, constants

# Streamlit setup
st.set_page_config(page_title="Clinical Diagnosis Support", layout="centered")
st.title("🩺 Clinical Diagnosis Support System")

# Input options: file OR text
st.markdown("Upload a lab report **or** type symptoms directly.")
uploaded_file = st.file_uploader(
    "Upload patient data (Text, Image, or Lab Report PDF)",
    type=["txt", "pdf", "jpg", "png"],
)
user_prompt = st.text_area(
    "Or describe symptoms manually",
    placeholder="E.g., Patient has high fever, dry cough for 3 days...",
)

# Proceed if either input is given
if uploaded_file or user_prompt.strip():
    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_type = detect_file_type(uploaded_file.name)

        with st.spinner("🔍 Extracting patient data..."):
            content = extract_content(file_bytes, file_type)
    else:
        content = user_prompt.strip()

    # LangChain tools
    tools = [
        pubmed_a_gemma_tool.pubmedgpt_tool,
        summarizer_tool.summarizer_tool,
    ]

    # LangChain LLM with Gemini
    llm = ChatGoogleGenerativeAI(
        model=constants.GEMIN25_PRO,
        temperature=0.7,
        google_api_key=variables.GOOGLE_AI_API_KEY,
    )

    # LangChain Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="chat-zero-shot-react-description",
        verbose=True,
    )

    with st.spinner("🤖 Generating diagnosis summary..."):
        result = agent.run(
            f"""Analyze the following patient data and 
            generate diagnostic summary:\n{content}"""
        )

    st.success("✅ Diagnostic Summary Ready")
    st.markdown("### 📝 Summary Report")
    st.text_area("Summary", value=result, height=300)
    st.download_button(
        "Download Summary", data=result, file_name="diagnosis_summary.txt"
    )

else:
    info = """📥 Please upload a file or 
    enter a symptom description to continue."""
    st.info(info)
