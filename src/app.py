import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from contextlib import asynccontextmanager

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

# Schéma de données entrée
class TextRequest(BaseModel):
    text: str

# Endpoint de prédiction
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
            "text": request.text,
            "prediction_class_id": int(prediction[0]),
            # On pourrait mapper l'ID vers un nom de classe si on avait le dictionaire correspondant
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": 'model' in artifacts}
