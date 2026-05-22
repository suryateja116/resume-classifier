from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import os
import re
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Load model & vectorizer ───────────────────────────────────
svm   = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl",  "rb"))

# ── Skill taxonomy ────────────────────────────────────────────
SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "r", "scala", "kotlin",
        "swift", "go", "rust", "typescript", "php", "ruby", "matlab"
    ],
    "Web & Frontend": [
        "html", "css", "javascript", "react", "angular", "vue",
        "tailwind", "bootstrap", "next.js", "graphql", "rest api"
    ],
    "Data & AI": [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "data analysis", "statistics", "tableau", "power bi",
        "big data", "spark", "hadoop", "data pipeline", "etl"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "google cloud", "docker", "kubernetes",
        "ci/cd", "terraform", "jenkins", "git", "linux", "bash"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis",
        "elasticsearch", "oracle", "sqlite", "nosql"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "agile", "scrum", "time management",
        "critical thinking", "presentation"
    ],
    "Domain": [
        "finance", "healthcare", "marketing", "sales",
        "human resources", "education", "supply chain", "cybersecurity"
    ]
}

# ── Job roles with rich descriptions ─────────────────────────
JOB_ROLES = [
    "Software Engineering",
    "Data Science & AI",
    "Information Technology",
    "Banking & Finance",
    "Human Resources",
    "Education & Teaching",
    "Sales & Marketing",
    "Healthcare",
    "Cybersecurity",
    "Product Management",
]

JOB_DESCRIPTIONS = [
    # Software Engineering
    "software development programming algorithms object oriented design "
    "system architecture debugging testing code review agile git python java c++",
    # Data Science & AI
    "machine learning deep learning data analysis python statistics "
    "tensorflow pytorch model training neural networks nlp computer vision pandas numpy",
    # Information Technology
    "it support systems administration networking cloud infrastructure "
    "aws azure linux windows active directory troubleshooting cybersecurity",
    # Banking & Finance
    "finance accounting banking investment portfolio risk management "
    "financial modelling excel bloomberg trading compliance audit",
    # Human Resources
    "recruitment talent acquisition hr management onboarding payroll "
    "employee relations performance management training development",
    # Education & Teaching
    "teaching curriculum lesson plan classroom education training "
    "assessment pedagogy e-learning mentoring university college",
    # Sales & Marketing
    "sales business development marketing crm lead generation "
    "digital marketing seo social media branding content strategy",
    # Healthcare
    "medical healthcare clinical nursing doctor patient care "
    "hospital diagnosis treatment pharmacology public health",
    # Cybersecurity
    "cybersecurity penetration testing ethical hacking network security "
    "firewall siem vulnerability assessment soc compliance iso",
    # Product Management
    "product management roadmap user stories stakeholder agile scrum "
    "market research wireframe product lifecycle kpi ux strategy",
]

job_vectors = tfidf.transform(JOB_DESCRIPTIONS)


# ── Helpers ───────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join(text.split())


def extract_text(filepath: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT."""
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'pdf':
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + " "
        return text

    if ext == 'docx':
        doc = docx.Document(filepath)
        return " ".join(p.text for p in doc.paragraphs)

    if ext == 'txt':
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()

    return ""


def extract_skills(text: str) -> dict:
    """Return skills grouped by category."""
    text_lower = text.lower()
    found = {}
    for category, skills in SKILL_CATEGORIES.items():
        matched = [s.title() for s in skills if s in text_lower]
        if matched:
            found[category] = matched
    return found


def extract_experience_years(text: str) -> int | None:
    """Detect years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+experience',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_education(text: str) -> list[str]:
    """Detect degree mentions."""
    degrees = [
        "phd", "ph.d", "doctorate",
        "master", "m.sc", "mba", "m.eng",
        "bachelor", "b.sc", "b.eng", "b.tech",
        "diploma", "associate"
    ]
    found = []
    text_lower = text.lower()
    for d in degrees:
        if d in text_lower:
            found.append(d.upper())
    # De-duplicate while preserving order
    seen = set()
    return [x for x in found if not (x in seen or seen.add(x))]


def match_jobs(resume_text: str) -> list[tuple]:
    """Return top job matches with normalised percentage scores."""
    resume_vec = tfidf.transform([resume_text])
    similarity  = cosine_similarity(resume_vec, job_vectors)[0]

    # Normalise to 0–100 based on the max score in this result set
    max_score = max(similarity) if max(similarity) > 0 else 1
    normalised = [(role, round(float(score / max_score) * 100, 1))
                  for role, score in zip(JOB_ROLES, similarity)]

    return sorted(normalised, key=lambda x: x[1], reverse=True)[:5]


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('resume')

    # Validation
    if not file or file.filename == '':
        flash("Please upload a resume file.")
        return redirect(url_for('home'))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        return redirect(url_for('home'))

    filename  = secure_filename(file.filename)
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        raw_text = extract_text(filepath)
    except Exception as e:
        flash(f"Could not read the file: {e}")
        return redirect(url_for('home'))
    finally:
        # Clean up uploaded file immediately after reading
        if os.path.exists(filepath):
            os.remove(filepath)

    if not raw_text.strip():
        flash("The file appears to be empty or unreadable.")
        return redirect(url_for('home'))

    cleaned = clean_text(raw_text)

    matches     = match_jobs(cleaned)
    predicted   = matches[0][0]
    skills      = extract_skills(raw_text)           # use raw for better matching
    exp_years   = extract_experience_years(raw_text)
    education   = extract_education(raw_text)

    return render_template(
        "result.html",
        role=predicted,
        matches=matches,
        skills=skills,             # now a dict: {category: [skill, ...]}
        exp_years=exp_years,
        education=education,
    )


@app.errorhandler(413)
def too_large(e):
    flash("File is too large. Maximum allowed size is 5 MB.")
    return redirect(url_for('home')), 413


if __name__ == '__main__':
    app.run(debug=True)