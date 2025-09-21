def generate_radiology_prompt(content: str, modality: str = "MRI") -> str:
    """
    Generates a structured prompt for MedGemma to produce a radiology report.

    :param content: Text or description of the image
    :param modality: Imaging modality, e.g., MRI, CT, X-ray
    :return: Formatted prompt
    """
    prompt = f"""
You are an expert radiologist. Generate a structured {modality} report for the patient data provided below.
The report must include the following sections:

1. **Findings**: Describe all abnormal and normal observations in detail.
2. **Impressions**: Provide a concise summary with diagnostic conclusions or recommendations.

Ensure:
- Professional radiology language
- Clear separation of Findings and Impressions
- Avoid unnecessary repetition
- Include relevant measurements, laterality, or location if mentioned

Patient data / Image description:
{content}

Generate the report in the format:

Findings:
<detailed findings here>

Impressions:
<summary / impression here>
"""
    return prompt
