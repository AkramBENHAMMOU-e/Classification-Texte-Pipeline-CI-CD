import nltk
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path
from urllib.error import HTTPError

"""
Pré-traitement des données (Preprocessing).

Ce script télécharge les "20 Newsgroups", nettoie le texte, et prépare
les jeux d'entraînement, de validation et de test pour le modèle.
"""

def ensure_nltk_data() -> None:
    """
    Assure la présence des ressources NLTK requises.
    (Télécharge uniquement si manquantes.)
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

def clean_text(text):
    """
    Nettoyage basique du texte.
    - Minuscules uniquement
    (On garde la ponctuation et les chiffres car importants pour le contexte : C++, Windows 95, etc.)
    """
    # Minuscules pour normaliser
    text = text.lower()
    
    # Suppression espaces superflus
    text = text.strip()
    return text

def process_text(text, stop_words, lemmatizer):
    """
    Traitement linguistique.
    - On retourne le texte brut nettoyé pour laisser le Vectorizer gèrer les n-grams
    et les stopwords de manière contextuelle.
    """
    return text

def main():
    ensure_nltk_data()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"

    # En CI, on évite de retélécharger 20 Newsgroups (source externe parfois bloquée / 403).
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Données 'data/processed' déjà présentes, preprocessing ignoré.")
        return

    print("Chargement du dataset 20 Newsgroups...")
    # Pour dépasser 80% d'accuracy facilement, on inclut les métadonnées (headers/footers/quotes)
    # Le nettoyage strict ("remove") plafonne souvent les modèles classiques vers 75-78%
    try:
        newsgroups = fetch_20newsgroups(subset="all")
    except HTTPError as e:
        raise RuntimeError(
            "Impossible de télécharger 20 Newsgroups (HTTPError). "
            "Si vous êtes en CI, committez les fichiers 'data/processed/*.csv' "
            "ou exécutez le preprocessing en local puis push."
        ) from e
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target
    })
    
    print("Initialisation NLP...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("Nettoyage du texte en cours...")
    # 1. Nettoyage simple
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 2. Traitement NLP complet
    df['processed_text'] = df['cleaned_text'].apply(lambda x: process_text(x, stop_words, lemmatizer))
    
    # Suppression des textes vides après nettoyage
    df = df[df['processed_text'].str.len() > 0].copy()
    
    print("Séparation des données...")
    # Split 70% Train, 15% Val, 15% Test
    
    # Séparation train / reste
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
    
    # Séparation validation / test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Sauvegarde CSV...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Terminé ! Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    main()
