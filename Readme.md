# Resume Classification and Career Recommendation System

## Overview
This project is an AI-powered Resume Classification System that analyzes resumes and predicts the most suitable job roles. It also extracts skills, identifies educational background, and provides job match scores.

The system combines Natural Language Processing (NLP), Machine Learning, and a web-based interface to deliver real-time insights.

---

## Features
- Upload resumes in PDF, DOCX, or TXT format
- Automatic text extraction
- Resume classification using Machine Learning (SVM)
- Skill extraction categorized by domain
- Job role matching using cosine similarity
- Detection of experience and education
- Interactive web-based UI using Flask

---

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- PDFPlumber (PDF extraction)
- python-docx (DOCX extraction)
- HTML, CSS, JavaScript

---

## Project Structure
project/
│
├── app.py
├── best_model.pkl
├── vectorizer.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── uploads/
├── Resume.csv
└── README.md

---

## Installation and Setup

### Step 1: Download the Project
Download and extract the project folder.

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run the Application
python app.py

### Step 4: Open in Browser
http://127.0.0.1:5000

---

## Model Details
- Algorithm: Support Vector Machine (SVM)
- Feature Extraction: TF-IDF Vectorization
- Problem Type: Multi-class classification
- Accuracy: ~66%

---

## Workflow
1. User uploads resume
2. System extracts text
3. Text is cleaned and processed
4. TF-IDF converts text into vectors
5. SVM predicts job role
6. Cosine similarity finds best job matches
7. Skills and other details are extracted
8. Results displayed on UI

---

## Limitations
- Moderate accuracy (~66%)
- Keyword-based skill extraction
- Limited dataset
- No deep learning models used

---

## Future Improvements
- Use BERT / Transformers for better NLP
- Improve dataset size and quality
- Add real-time learning
- Integrate with job portals
- Add explainable AI features

---

## Author
Surya Teja

---

## License
This project is developed for academic purposes.