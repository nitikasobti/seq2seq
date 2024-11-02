# pdf_extractor.py
import PyPDF2
import re

from data_preprocessing import clean_text

def extract_text_from_pdf(file):
    """
    Extracts and cleans text from a PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt("")  # Attempt to decrypt without password
            except Exception as e:
                return f"Failed to decrypt PDF: {e}"
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        cleaned_text = clean_extracted_text(text)
        return cleaned_text
    
    except Exception as e:
        return f"An error occurred while extracting text: {e}"

def clean_extracted_text(text):
    """
    Cleans the extracted text by removing extra spaces and newline characters.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
