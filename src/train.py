import argparse
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix

def main(run_name=None):
    # Configuration MLflow (support local ou distant, cohérent avec train_old.py)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Text_Classification_Projet9")

    # Nom du run MLflow (CLI > env > défaut MLflow)
    if run_name is None:
        run_name = os.getenv("MLFLOW_RUN_NAME")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")

    print("Chargement des données nettoyées...")
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
    except FileNotFoundError:
        print("Erreur: Fichiers introuvables. Lancez 'src/preprocess.py' d'abord.")
        return

    train_df = train_df.dropna(subset=['processed_text'])
    test_df = test_df.dropna(subset=['processed_text'])

    # Paramètres du modèle / représentation du texte
    MAX_FEATURES = 50000
    # Utilisation d'un seul modèle : LinearSVC (le plus performant pour la classification de texte)
    NGRAM_RANGE = (1, 2)
    MIN_DF = 2
    MAX_DF = 0.5
    SUBLINEAR_TF = True

    # Paramètres SVM / calibration (pour suivi MLflow)
    SVM_C_VALUES = [1.0]
    CV_FOLDS = 3
    
    # Démarrage du run MLflow
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log de paramètres alignés avec train_old.py
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("ngram_range", f"{NGRAM_RANGE[0]}_{NGRAM_RANGE[1]}")
        mlflow.log_param("min_df", MIN_DF)
        mlflow.log_param("max_df", MAX_DF)
        mlflow.log_param("sublinear_tf", SUBLINEAR_TF)
        mlflow.log_param("model_type", "LinearSVC")
        mlflow.log_param("svm_C_candidates", ",".join(map(str, SVM_C_VALUES)))
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("best_C", SVM_C_VALUES[0])
        
        print("Vectorisation (TF-IDF)...")
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE, 
            sublinear_tf=SUBLINEAR_TF, 
            min_df=MIN_DF,
            max_df=MAX_DF
        )
        X_train = vectorizer.fit_transform(train_df['processed_text'])
        X_test = vectorizer.transform(test_df['processed_text'])
        
        y_train = train_df['target']
        y_test = test_df['target']
        
        print(f"Dimensionnalité: {X_train.shape}")

        print("Entraînement (LinearSVC avec calibration pour obtenir les probabilités)...")
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        
        # LinearSVC n'a pas de predict_proba, on utilise CalibratedClassifierCV pour obtenir les probabilités
        base_model = LinearSVC(random_state=42, C=SVM_C_VALUES[0], max_iter=3000)
        model = CalibratedClassifierCV(base_model, cv=CV_FOLDS)
        model.fit(X_train, y_train)

        print("Évaluation...")
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted): {recall:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        
        # Log des métriques dans MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Rapport détaillé
        print("\nRapport détaillé :")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Sauvegarde des résultats locaux (Pour CML)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

        # 1. Sauvegarde des métriques en JSON
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        # 2. Sauvegarde du rapport texte
        with open("reports/classification_report.txt", "w") as f:
            f.write(report)
            
        # 3. Génération et sauvegarde de la matrice de confusion
        print("Génération de la matrice de confusion...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        plt.savefig('reports/confusion_matrix.png')
        plt.close()
        
        # Log de la matrice de confusion dans MLflow
        mlflow.log_artifact('reports/confusion_matrix.png')

        # 4. Sauvegarde modèle et vectorizer (Local + MLflow)
        joblib.dump(model, 'models/model.joblib')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact('models/tfidf_vectorizer.joblib')
        
        print("Terminé ! Modèle, métriques, rapport et matrice de confusion sauvegardés (Local + MLflow).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text classification model with MLflow tracking.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nom du run MLflow (sinon utilise MLFLOW_RUN_NAME ou le nom par défaut).",
    )
    args = parser.parse_args()
    main(run_name=args.run_name)
