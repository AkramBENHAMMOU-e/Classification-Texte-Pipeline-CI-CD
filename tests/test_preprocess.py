import sys
import os
import pytest
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ajout du dossier root au path pour importer les modules src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text, process_text

@pytest.fixture
def lemmatizer():
    return WordNetLemmatizer()

@pytest.fixture
def stop_words():
    return set(stopwords.words('english'))

def test_clean_text_lowercase():
    """Vérifie que le texte est bien mis en minuscule."""
    raw = "Hello World"
    assert clean_text(raw) == "hello world"

def test_clean_text_keeps_punctuation_and_numbers():
    """Vérifie que la ponctuation et les chiffres sont conservés (important pour C++, Win95, etc.)."""
    raw = "Hello, world! 123"
    # Le nouveau preprocessing garde la ponctuation et les nombres
    assert clean_text(raw).strip() == "hello, world! 123"

def test_process_text_passthrough(lemmatizer, stop_words):
    """Vérifie que process_text retourne le texte tel quel (TF-IDF gère le reste)."""
    raw_cleaned = "cats are running"
    # Le nouveau preprocessing retourne le texte sans modification
    result = process_text(raw_cleaned, stop_words, lemmatizer)
    assert result == "cats are running"

def test_process_text_keeps_all_words(lemmatizer, stop_words):
    """Vérifie que process_text conserve tous les mots (TF-IDF gère les stopwords)."""
    raw_cleaned = "this is a test"
    # Le nouveau preprocessing retourne le texte sans modification
    result = process_text(raw_cleaned, stop_words, lemmatizer)
    assert "this" in result
    assert "is" in result
    assert "test" in result
