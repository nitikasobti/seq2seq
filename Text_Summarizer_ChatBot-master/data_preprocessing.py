# import PyPDF2
# import re

# def extract_text_from_pdf(file):
#     try:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
        
#         if pdf_reader.is_encrypted:
#             try:
#                 pdf_reader.decrypt("qwerty")
#             except Exception as e:
#                 return f"Failed to decrypt PDF: {e}"
        
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text

#         cleaned_text = clean_extracted_text(text)
#         return cleaned_text
#     except Exception as e:
#         return f"An error occurred while extracting text: {e}"

# def clean_extracted_text(text):
#     text = text.replace('\n', ' ')
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     return text
# 
# 
# 
# 
# data_preprocessing.py
# import re
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize

# # Download required NLTK data
# nltk.download("punkt")

# def clean_text(text):
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip().lower()
#     return text

# def tokenize_text(text):
#     return word_tokenize(clean_text(text))

# def preprocess_dataset(dataset):
#     tokenized_data = []
#     for text in dataset:
#         tokenized_text = tokenize_text(text)
#         tokenized_data.append(tokenized_text)
#     return tokenized_data
# data_preprocessing.py
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data only once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Example vocabulary mapping (you should define this based on your dataset)
vocab = {'your': 1, 'input': 2, 'text': 3, 'here': 4, 'another': 5}  # Extend this

def clean_text(text):
    # Clean and normalize the text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

def tokenize_text(text):
    # Tokenize the text and convert to IDs based on vocab
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    token_ids = [vocab[token] for token in tokens if token in vocab]  # Convert to IDs, filter OOV
    return token_ids

def preprocess_dataset(dataset):
    # Preprocess each text in the dataset
    tokenized_data = []
    for text in dataset:
        tokenized_text = tokenize_text(text)
        tokenized_data.append(tokenized_text)
    return tokenized_data
