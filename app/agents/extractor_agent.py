from app.utils.image_parser import extract_text_from_image
from app.utils.document_parser import extract_text_from_pdf
from app.utils.text_cleaner import clean_text

def extract_content(raw_data, file_type):
    if file_type == "pdf":
        return clean_text(extract_text_from_pdf(raw_data))
    elif file_type == "image":
        return clean_text(extract_text_from_image(raw_data))
    else:
        return clean_text(raw_data.decode("utf-8"))
