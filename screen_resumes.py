import os
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def get_similarity(resume_text, job_description):
    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([job_description])
    similarity = cosine_similarity(resume_embedding, jd_embedding)
    return similarity[0][0]

job_description ="""
In this role, you will work with cross functional teams ranging from PON, Ethernet switching and routing software, subscriber management for Triple Play services, performance monitoring to network management to build, integrate, unit test and support software modules within a large-scale system.
For the position of new trainee engineer, we are looking for a well-organized graduate to join our team. Meeting attendance, desktop and field research, all practical and administrative tasks assigned by the supervisor, site visits when necessary, project assistance, improvement suggestions, and report writing are among the duties of the trainee engineer. You should be willing to venture outside of your comfort zone and have the ability to collaborate with others..
"""



resume_folder = "resumes"
scores = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        path = os.path.join(resume_folder, file)
        resume_text = extract_text_from_pdf(path)
        similarity = get_similarity(resume_text, job_description)
        scores.append((file, similarity))

sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
print("\nTop Matching Resumes:")
for file, score in sorted_scores:
    print(f"{file} --> Match Score: {score:.4f}")
