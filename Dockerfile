# Image de base légère avec Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances en premier (pour le cache Docker)
COPY requirements.txt .
COPY models/ ./models/
COPY data/processed/ ./data/processed/


# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Exposer le port de l'API
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
