import pdfplumber
import io

def extract_text_from_pdf(pdf_bytes):
    text = ''
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text
