# app.py
import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Keep your original imports
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.multimodel_agent import MedicalImageModel
from app.config import variables, constants, auth # app.
from app.tools import summarizer_tool, pubmed_a_gemma_tool
from app.utils.modality_utils import detect_modality_with_llm
from app.utils.pdf_utils import markdown_to_pdf
from app.utils.prompt_utils import generate_radiology_prompt

# Initialize logging
logging.basicConfig(level=logging.INFO, format=constants.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Run any required auth/login from your original app
try:
    auth.hf_login()
    logger.info("HF login complete.")
    MedicalImageModel()
    logger.info("Loaded the Med Gemma Model")
except Exception as e:
    # don't crash the app if login fails; surface the error on requests
    logger.warning("HF login failed during startup: %s", e)

# FastAPI setup
app = FastAPI(title="Clinical Diagnosis Support — Multi-Modal SUDSS")

# Allow CORS if you plan to call from a browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnoseRequest(BaseModel):
    text: Optional[str] = None


def save_uploadfile_tmp(upload_file: UploadFile) -> str:
    """
    Save UploadFile to the present working directory and return the saved path.
    Caller should remove the file manually if needed.
    """
    # Get the present working directory (where the app is running)
    pwd = os.getcwd()

    # Construct the full path to save the file
    save_path = os.path.join(pwd, upload_file.filename)
    # Save the uploaded file contents to the PWD
    with open(save_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)

    return save_path


def build_agent_and_tools(image_path: Optional[str] = None):
    """
    Returns an initialized agent similar to your Streamlit flow.
    Make sure variables.GOOGLE_AI_API_KEY and constants.GEMIN25_PRO are available.
    """
    # Initialize MedGemma Singleton (same as your streamlit code)
    medgemma_model = MedicalImageModel()

    def medgemma_tool_func(content: str):
        """
        LangChain callable function for MedGemma inference.
        Accepts any content (image description or text) and returns structured report.
        """
        report = medgemma_model.run_inference(image_path=image_path, prompt=content)
        logger.info("MedGemma Report Generated.")
        return report

    medgemma_tool = Tool(
        name="MedGemma Summarizer",
        func=medgemma_tool_func,
        description=(
            "Generate a structured radiology report with Findings and Impressions "
            "from patient data (accepts only medical image)."
        ),
    )

    # Compose tool list
    tools = [
        medgemma_tool,
        pubmed_a_gemma_tool.pubmedgpt_tool,
        summarizer_tool.summarizer_tool,
    ]

    # Master LLM
    master_agent = ChatGoogleGenerativeAI(
        model=constants.GEMIN25_PRO,
        temperature=0.7,
        google_api_key=variables.GOOGLE_AI_API_KEY,
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=master_agent,
        agent_type="chat-zero-shot-react-description",
        verbose=True,
    )

    return agent


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/diagnose")
async def diagnose(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Accept either a file upload (txt, pdf, jpg, png) or a text description (form field `text`).
    Returns JSON containing the diagnostic summary result.
    """
    # Validate input: require at least one
    if not file and (not text or not text.strip()):
        raise HTTPException(status_code=400, detail="Provide either `file` or `text` input.")

    temp_path = None
    raw_content = None

    try:
        # If a file was provided, prefer it over the text (mirrors your Streamlit logic)
        if file:
            temp_path = save_uploadfile_tmp(file)
            # We will pass a reference to the agent like your previous code did
            raw_content = f"[Image file at {temp_path}]"
            logger.info("File saved to %s", temp_path)
        else:
            raw_content = text.strip()

        # Detect modality
        image_path = temp_path if temp_path else None
        try:
            modality = detect_modality_with_llm("", image_path=image_path)
            logger.info("Detected modality: %s", modality)
        except Exception as e:
            # Don't fail hard on modality detection; keep a fallback
            modality = None
            logger.warning("Modality detection failed: %s", e)

        # Generate structured prompt
        prompt = generate_radiology_prompt(raw_content, modality)

        # Build agent (this will re-create MedGemma and agent; if heavy you can optimize later)
        agent = build_agent_and_tools(image_path=image_path)

        # Build refined prompt (same text as streamlit version)
        refined_prompt = (
            "You are a clinical AI assistant. You have access to the MedGemma tool to "
            "generate structured radiology reports.\n"
            "Using the following patient data, produce a detailed diagnostic summary that "
            "includes:\n\n"
            "1. A concise overview of findings\n"
            "2. Critical impressions and recommendations\n"
            "3. Reference relevant evidence if applicable\n"
            "4. Maintain professional medical terminology\n\n"
            f"{prompt}\n\n"
            "Use the MedGemma Summarizer tool wherever appropriate.\n"
        )

        # Run the agent synchronously (initialize_agent returns a sync agent in your original code)
        logger.info("Running agent...")
        result = agent.invoke(refined_prompt)

        return JSONResponse(status_code=200, content={"summary": result})

    except Exception as e:
        logger.exception("Error during diagnosis: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    finally:
        # cleanup any temp file created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Removed temp file %s", temp_path)
            except Exception:
                logger.warning("Failed to remove temp file %s", temp_path)


@app.post("/diagnose/download")
async def diagnose_download(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Same as /diagnose but returns a PDF file (diagnosis_summary.pdf) as streaming response.
    """
    # Validate input
    if not file and (not text or not text.strip()):
        raise HTTPException(status_code=400, detail="Provide either `file` or `text` input.")

    temp_path = None
    raw_content = None

    try:
        if file:
            temp_path = save_uploadfile_tmp(file)
            raw_content = f"[Image file at {temp_path}]"
        else:
            raw_content = text.strip()

        # Detect modality
        image_path = temp_path if temp_path else None
        try:
            modality = detect_modality_with_llm("", image_path=image_path)
            logger.info("Detected modality: %s", modality)
        except Exception as e:
            modality = None
            logger.warning("Modality detection failed: %s", e)

        # Generate structured prompt
        prompt = generate_radiology_prompt(raw_content, modality)

        # Build agent and run
        agent = build_agent_and_tools(image_path=image_path)
        refined_prompt = (
            "You are a clinical AI assistant. You have access to the MedGemma tool to "
            "generate structured radiology reports.\n"
            "Using the following patient data, produce a detailed diagnostic summary that "
            "includes:\n\n"
            "1. A concise overview of findings\n"
            "2. Critical impressions and recommendations\n"
            "3. Reference relevant evidence if applicable\n"
            "4. Maintain professional medical terminology\n\n"
            f"{prompt}\n\n"
            "Use the MedGemma Summarizer tool wherever appropriate.\n"
        )

        logger.info("Running agent to produce summary for PDF...")
        result = agent.invoke(refined_prompt)

        # Convert to PDF — assumes markdown_to_pdf returns bytes of PDF
        try:
            pdf_bytes = markdown_to_pdf(result)
            if not isinstance(pdf_bytes, (bytes, bytearray)):
                raise TypeError("markdown_to_pdf did not return bytes.")
        except Exception as e:
            logger.exception("Failed to generate PDF: %s", e)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

        # Return as downloadable streaming response
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="diagnosis_summary.pdf"'},
        )

    except Exception as e:
        logger.exception("Error during diagnosis download: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Removed temp file %s", temp_path)
            except Exception:
                logger.warning("Failed to remove temp file %s", temp_path)
