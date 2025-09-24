import pytesseract
from PIL import Image
import io
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_from_pdf(pdf_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(pdf_bytes))
    ocr_text = pytesseract.image_to_string(img, lang='eng')
    doc = nlp(ocr_text)
    extracted = {
        "claimant_name": "",
        "village": "",
        "survey_no": "",
        "date": "",
        "confidence": 0.85,
        "raw_text": ocr_text
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted["claimant_name"] = ent.text
        elif ent.label_ == "GPE":
            extracted["village"] = ent.text
        elif ent.label_ == "DATE":
            extracted["date"] = ent.text
    return extracted