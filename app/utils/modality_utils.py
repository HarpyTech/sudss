from app.agents.multimodel_agent import MedicalImageModel


def detect_modality_with_llm(content: str, image_path: str = None) -> str:
    """
    Detect the modality of the given input (MRI, X-ray, CT, Ultrasound, or Medical Text)
    using MedGemma / LLM inference.
    """
    # Create a lightweight MedGemma instance (singleton)
    medgemma = MedicalImageModel()

    # Prompt LLM to detect modality
    prompt = f"""
You are a radiology expert. Identify the imaging modality or input type 
from the following patient data. Possible outputs: MRI, CT, X-ray, Ultrasound, Medical Text.
Respond with only one of these options.

Patient data / description:
{content}
"""

    # Run inference
    modality = medgemma.run_inference(image_path=image_path, prompt=prompt)

    # Clean output
    modality = modality.strip().split("\n")[0]  # get first line
    valid_modalities = ["MRI", "CT", "X-ray", "Ultrasound", "Medical Text"]
    if modality not in valid_modalities:
        modality = "Medical Image"
    return modality
