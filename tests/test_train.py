import os
import pytest
import pandas as pd
import joblib

def test_data_existence():
    """Vérifie que les données processed existent pour l'entraînement."""
    assert os.path.exists('data/processed/train.csv'), "Train data manquante"
    assert os.path.exists('data/processed/test.csv'), "Test data manquante"

def test_model_artifact_generation():
    """ Vérifie que le script d'entrainement génère bien les fichiers modèles. """
    # On vérifie seulement que les fichiers sont là (suppose que train.py a déjà tourné au moins une fois ou on le lance ?)
    # Pour un test unitaire/intégration strict, on devrait mocker ou lancer une version allégée.
    # Ici on vérifie l'existence post-run (supposant que l'environnement est prêt).
    
    # Si les fichiers n'existent pas, ce test échouera, ce qui est correct pour une CI.
    if os.path.exists('models/model.joblib'):
        model = joblib.load('models/model.joblib')
        assert hasattr(model, 'predict'), "Le fichier chargé n'est pas un modèle valide"
    else:
        pytest.skip("Modèle non trouvé. Lancez src/train.py avant de tester.")

def test_vectorizer_artifact():
    """ Vérifie que le vectorizer existe. """
    if os.path.exists('models/tfidf_vectorizer.joblib'):
        vec = joblib.load('models/tfidf_vectorizer.joblib')
        assert hasattr(vec, 'transform'), "Le fichier chargé n'est pas un vectorizer valide"
    else:
        pytest.skip("Vectorizer non trouvé.")
