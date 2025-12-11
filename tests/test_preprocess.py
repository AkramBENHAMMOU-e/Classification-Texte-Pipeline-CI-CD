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

def test_clean_text_remove_punctuation():
    """Vérifie suppression ponctuation et chiffres."""
    raw = "Hello, world! 123"
    expected = "hello world " # L'espace final peut dépendre de l'implémentation du regex
    # On strip pour être sûr
    assert clean_text(raw).strip() == "hello world"

def test_process_text_lemmatization(lemmatizer, stop_words):
    """Vérifie la lemmatisation standard."""
    raw_cleaned = "cats are running"
    # cats -> cat, are -> be (selon modèle wordnet mais 'are' est often stopword), running -> running (adj) ou run (v)
    # process_text fait aussi le retrait des stop words
    # 'are' est un stop word
    result = process_text(raw_cleaned, stop_words, lemmatizer)
    assert "cat" in result
    assert "run" in result or "running" in result 

def test_process_text_stopwords(lemmatizer, stop_words):
    """Vérifie que les stop words sont retirés."""
    raw_cleaned = "this is a test"
    # this, is, a -> stopwords
    # test -> test
    result = process_text(raw_cleaned, stop_words, lemmatizer)
    assert "this" not in result
    assert "is" not in result
    assert "test" in result
