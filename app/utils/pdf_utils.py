import io
from fpdf import FPDF
import re


def markdown_to_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Default font
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        # Heading (start with #)
        if line.startswith("### "):
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 8, line.replace("### ", ""))
            pdf.set_font("Arial", size=12)
        elif line.startswith("## "):
            pdf.set_font("Arial", "B", 16)
            pdf.multi_cell(0, 10, line.replace("## ", ""))
            pdf.set_font("Arial", size=12)
        elif line.startswith("# "):
            pdf.set_font("Arial", "B", 18)
            pdf.multi_cell(0, 12, line.replace("# ", ""))
            pdf.set_font("Arial", size=12)
        # Bold (**text**)
        line = re.sub(r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), line)
        pdf.multi_cell(0, 8, line)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_bytes = pdf_output.getvalue()
    pdf_output.close()
    return pdf_bytes
