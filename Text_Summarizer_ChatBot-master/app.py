# import streamlit as st
# from pdf_extractor import extract_text_from_pdf, clean_extracted_text
# from textrank import textrank_summarizer
# import base64

# st.set_page_config(
#     page_title="Portable Document Analyser",
#     page_icon=":sparkles:",
#     layout="wide"
# )

# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# logo_path = "new_logo.jpeg"
# logo_base64 = get_base64_of_bin_file(logo_path)

# st.markdown(f"""
#     <style>
#         /* Your CSS Styling */
#     </style>
# """, unsafe_allow_html=True)

# st.markdown(f"""
#     <div class="header">
#         <h2>Portable Document Analyser</h2>
#         <img src="data:image/png;base64,{logo_base64}" class="logo">
#     </div>
#     <hr style="border-top: 2px solid #bb86fc;">
# """, unsafe_allow_html=True)

# st.markdown('<div class="title-section">Upload your PDF </div>', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file is not None:
#     try:
#         raw_text = extract_text_from_pdf(uploaded_file)
#         cleaned_text = clean_extracted_text(raw_text)

#         st.markdown('<div class="option-box">', unsafe_allow_html=True)
#         st.subheader("Extracted Text")
#         st.text_area("Extracted Text", cleaned_text, height=300)
#         st.markdown('</div>', unsafe_allow_html=True)

#         st.markdown('<div class="title-section">What would you like to do with the PDF?</div>', unsafe_allow_html=True)

#         st.markdown('<div class="option-box">', unsafe_allow_html=True)
#         st.subheader("Summarize PDF")

#         summary_format = st.radio("Select summary format:", ('Paragraph', 'Points'))
#         summary_percentage = st.slider("Select summary length (% of original text):", min_value=10, max_value=100, value=30)

#         if st.button("Summarize"):
#             with st.spinner("Summarizing... Please wait."):
#                 format_choice = 'paragraph' if summary_format == 'Paragraph' else 'points'
#                 summary = textrank_summarizer(cleaned_text, percentage=summary_percentage, format=format_choice)
#                 st.subheader("Summary")
#                 st.success(summary)

#         st.markdown('</div>', unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"Error processing the PDF file: {e}")

# st.markdown("""
#     <div class="footer">
#         <p>Portable Document Analyser</p>
#     </div>
# """, unsafe_allow_html=True)
# app.py
import streamlit as st
import torch
from data_preprocessing import clean_text, tokenize_text
from pdf_extractor import extract_text_from_pdf
from seq2seq_model import Encoder, Decoder, Seq2Seq

st.title("Text Summarization from Scratch")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.write("Original Text")
    st.text_area("Extracted Text", text, height=300)
    
    # Load model
    vocab_size = 10000  # Example; should match your training config
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers,dropout)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)

    st.write("Summarized Text")
    if st.button("Generate Summary"):
        with torch.no_grad():
            tokenized_input = tokenize_text(text)
        if not tokenized_input:
            st.error("Input text is empty or unrecognized. Please check your input.")
        else:
            # Convert to tensor and add batch dimension, ensuring it's of type Long
            tokens_tensor = torch.tensor(tokenized_input, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

            # Generate summary
            summary = model(tokens_tensor, tokens_tensor)  # You may need to adjust this call
            summary_text = ' '.join(map(str, summary.cpu().numpy()))  # Convert summary tensor back to text

            st.write(summary_text)
