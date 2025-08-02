def detect_file_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext in ["jpg", "jpeg", "png"]:
        return "image"
    elif ext == "pdf":
        return "pdf"
    else:
        return "text"
