import streamlit as st
import os
import fitz  # keep this as-is
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()

def calculate_similarity(resume_text, jd):
    resume_emb = model.encode([resume_text])
    jd_emb = model.encode([jd])
    score = cosine_similarity(resume_emb, jd_emb)[0][0]
    return score

st.title("Automated Resume Screening")
st.write("Upload resumes and enter a job description to get matching scores.")

# Input Job Description
job_description = st.text_area("Enter Job Description", height=200)

# Upload resumes
uploaded_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

# Process resumes
if st.button("Start Screening") and uploaded_files and job_description:
    results = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        score = calculate_similarity(text, job_description)
        results.append((file.name, score))
    
    # Sort and display
    results.sort(key=lambda x: x[1], reverse=True)
    st.subheader("Matching Results:")
    for name, score in results:
        st.write(f"**{name}** → Match Score: `{score:.2%}`")
