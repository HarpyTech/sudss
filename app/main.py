import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.multimodel_agent import MedicalImageModel
from config import variables, constants, auth
from tools import summarizer_tool, pubmed_a_gemma_tool
from utils.modality_utils import detect_modality_with_llm
from utils.pdf_utils import markdown_to_pdf
from utils.prompt_utils import generate_radiology_prompt


auth.hf_login()
# -------------------- Streamlit Setup -------------------- #
st.set_page_config(page_title="Clinical Diagnosis Support", layout="centered")
st.title("🩺 Multi-Modal: SUDSS")

st.markdown("Upload a lab report, image, or type symptoms directly.")
uploaded_file = st.file_uploader(
    "Upload patient data (Text, PDF, Image)", type=["txt", "pdf", "jpg", "png"]
)
user_prompt = st.text_area(
    "Or describe symptoms manually",
    placeholder="E.g., Patient has high fever, dry cough for 3 days...",
)

# -------------------- Process Input -------------------- #
raw_content = None
temp_path = None

if uploaded_file and user_prompt.strip():
    if uploaded_file:
        st.info("Loading the uploaded file and ignoring manual input.")
        file_bytes = uploaded_file.read()

        temp_path = f"./{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        raw_content = f"[Image file at {temp_path}]"

    elif user_prompt.strip():
        raw_content = user_prompt.strip()

    if raw_content:
        st.info("Processing input data...")
        # For images, reuse temp_path if content references it
        image_path = temp_path if temp_path else None

        # -------------------- Detect Modality via LLM -------------------- #
        modality = detect_modality_with_llm("", image_path=image_path)

        # -------------------- Generate Structured Prompt -------------------- #
        prompt = generate_radiology_prompt(raw_content, modality)

        # -------------------- Initialize MedGemma Singleton -------------------- #
        medgemma_model = MedicalImageModel()

        # -------------------- Define MedGemma as LangChain Tool -------------------- #
        def medgemma_tool_func(content: str):
            """
            LangChain callable function for MedGemma inference.
            Accepts any content (image description or text) and returns structured report.
            """

            report = medgemma_model.run_inference(image_path=image_path, prompt=content)

            print("MedGemma Report Generated.")

            return report

        medgemma_tool = Tool(
            name="MedGemma Summarizer",
            func=medgemma_tool_func,
            description=(
                "Generate a structured radiology report with Findings and Impressions "
                "from patient data (accepts only medical image)."
            ),
        )

        # -------------------- LangChain Tools -------------------- #
        tools = [
            medgemma_tool,  # MedGemma integrated
            pubmed_a_gemma_tool.pubmedgpt_tool,
            summarizer_tool.summarizer_tool,
        ]

        # -------------------- Master LLM -------------------- #
        master_agent = ChatGoogleGenerativeAI(
            model=constants.GEMIN25_PRO,
            temperature=0.7,
            google_api_key=variables.GOOGLE_AI_API_KEY,
        )

        # -------------------- Initialize Agent -------------------- #
        agent = initialize_agent(
            tools=tools,
            llm=master_agent,
            agent_type="chat-zero-shot-react-description",
            verbose=True,
        )

        # -------------------- Refined Agent Prompt -------------------- #
        refined_prompt = (
            f"You are a clinical AI assistant. You have access to the MedGemma tool to "
            "generate structured radiology reports.\n"
            "Using the following patient data, produce a detailed diagnostic summary that "
            "includes:\n\n"
            "1. A concise overview of findings\n"
            "2. Critical impressions and recommendations\n"
            "3. Reference relevant evidence if applicable\n"
            "4. Maintain professional medical terminology\n\n"
            "Patient Data / Description:\n"
            f"{raw_content}\n\n"
            "Use the MedGemma Summarizer tool wherever appropriate.\n"
        )

        # -------------------- Generate Summary -------------------- #
        with st.spinner("🤖 Generating final diagnostic summary..."):
            result = agent.run(refined_prompt)

        # -------------------- Display Results -------------------- #
        st.success("✅ Diagnostic Summary Ready")
        st.markdown("### 📝 Final Clinical Summary")
        st.markdown(result)
        st.download_button(
            "Download Summary",
            data=markdown_to_pdf(result),
            file_name="diagnosis_summary.pdf",
            mime="application/pdf",
        )

else:
    st.info("📥 Please upload a file or enter a symptom description to continue.")
