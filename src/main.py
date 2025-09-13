# main.py
import os
import pandas as pd
from src.services.preprocessing import pipeline_preprocessing
from src.services.feature_engineering import feature_engineering
from src.services.train import pipeline_train
from src.services.evaluate import pipeline_evaluate
from src.utils.utils import upload_csv_to_s3

def main():
    # Paths dos arquivos
    MODEL_PATH = r"src\models\model.pkl"

    applicants_path = r"src/data/raw/applicants.json"
    prospects_path = r"src/data/raw/prospects.json"
    vagas_path = r"src/data/raw/vagas.json"

    preprocessed_csv = r"src/data/processed/preprocessed_data.csv"
    feature_engineered_csv = r"src/data/processed/feature_engineered_data.csv"

    # ---------------------------
    # 1) Pré-processamento
    # ---------------------------
    print("=== Iniciando Pré-processamento ===")
    df, encoders, scaler = pipeline_preprocessing(applicants_path, prospects_path, vagas_path)
    df.to_csv(preprocessed_csv, index=False, encoding="utf-8")
    upload_csv_to_s3(preprocessed_csv, "preprocessed_data.csv")
    print(f"CSV pré-processado salvo em: {preprocessed_csv}")

    # ---------------------------
    # 2) Feature Engineering
    # ---------------------------
    print("\n=== Iniciando Feature Engineering ===")
    df = pd.read_csv(preprocessed_csv, parse_dates=True)
    df = feature_engineering(df)
    df.to_csv(feature_engineered_csv, index=False, encoding="utf-8")
    upload_csv_to_s3(feature_engineered_csv, "feature_engineered_data.csv")
    print(f"CSV com features geradas salvo em: {feature_engineered_csv}")

    # ---------------------------
    # 3) Treinamento
    # ---------------------------
    print("\n=== Iniciando Treinamento ===")
    pipeline_train()

    # ---------------------------
    # 4) Validação
    # ---------------------------
    print("\n=== Iniciando Avaliações de Métricas===")
    pipeline_evaluate()

    # ---------------------------
    # 5) Subindo App Streamlit
    # ---------------------------
    #Executar no terminal streamlit run src/app/app.py


if __name__ == "__main__":
    main()