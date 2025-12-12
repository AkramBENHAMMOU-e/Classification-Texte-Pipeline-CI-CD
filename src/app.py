import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import sys
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from contextlib import asynccontextmanager

# Imports pour lecture de fichiers
from PyPDF2 import PdfReader
from docx import Document

# Calcul du chemin absolu du projet (parent du dossier src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.preprocess import clean_text, process_text

# Chargement des artefacts au démarrage
artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: chargement des artefacts
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        model_path = os.path.join(PROJECT_ROOT, 'models', 'model.joblib')
        vectorizer_path = os.path.join(PROJECT_ROOT, 'models', 'tfidf_vectorizer.joblib')
        
        artifacts['model'] = joblib.load(model_path)
        artifacts['vectorizer'] = joblib.load(vectorizer_path)
        artifacts['stop_words'] = set(stopwords.words('english'))
        artifacts['lemmatizer'] = WordNetLemmatizer()
        print(f"Artefacts chargés depuis {PROJECT_ROOT}")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
    
    yield  # L'app tourne ici
    
    # Shutdown (optionnel)
    artifacts.clear()

# Initialisation de l'app avec lifespan
app = FastAPI(
    title="Text Classification API",
    description="API pour classifier des articles de journaux (20 Newsgroups)",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS pour le frontend Angular
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schéma de données entrée
class TextRequest(BaseModel):
    text: str

# Fonction helper pour extraire le texte des fichiers
def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = file.file.read()
    
    if filename.endswith('.txt') or filename.endswith('.md'):
        return content.decode('utf-8')
    
    elif filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    else:
        raise ValueError(f"Format non supporté: {filename}")

# Endpoint de prédiction par texte
@app.post("/predict")
def predict(request: TextRequest):
    if 'model' not in artifacts or 'vectorizer' not in artifacts:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    
    try:
        # 1. Prétraitement
        cleaned_txt = clean_text(request.text)
        processed_txt = process_text(cleaned_txt, artifacts['stop_words'], artifacts['lemmatizer'])
        
        # 2. Vectorisation
        vectorized_txt = artifacts['vectorizer'].transform([processed_txt])
        
        # 3. Prédiction
        prediction = artifacts['model'].predict(vectorized_txt)
        
        # Retour
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "prediction_class_id": int(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de prédiction par fichier
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if 'model' not in artifacts or 'vectorizer' not in artifacts:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    
    try:
        # 1. Extraire le texte du fichier
        text = extract_text_from_file(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Le fichier est vide ou illisible.")
        
        # 2. Prétraitement
        cleaned_txt = clean_text(text)
        processed_txt = process_text(cleaned_txt, artifacts['stop_words'], artifacts['lemmatizer'])
        
        # 3. Vectorisation
        vectorized_txt = artifacts['vectorizer'].transform([processed_txt])
        
        # 4. Prédiction
        prediction = artifacts['model'].predict(vectorized_txt)
        
        return {
            "filename": file.filename,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "prediction_class_id": int(prediction[0]),
            "status": "success"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": 'model' in artifacts}

