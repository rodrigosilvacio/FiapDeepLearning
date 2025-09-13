import pandas as pd
import joblib
import os
import io
import boto3
from dotenv import load_dotenv
from typing import Any


# --- Constantes ---
PROCESSED_DATA_PATH = r"src/data/processed/feature_engineered_data.csv"
MODEL_PATH = r"src/models/model.pkl"

import os
import boto3

def upload_csv_to_s3(local_path: str, s3_key: str):
    """
    Faz upload de um arquivo CSV para o S3.
    local_path: caminho local do CSV
    s3_key: caminho/nome que o CSV terá dentro do bucket
    """
    load_dotenv()
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    region_name = os.getenv("AWS_REGION", "us-east-1")
    
    if not bucket_name:
        raise ValueError("⚠️ Variável de ambiente AWS_BUCKET_NAME não encontrada!")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name
    )
    
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"✅ CSV enviado para s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Erro ao realizar upload de arquivo para o S3: {e}")

def read_csv_s3(bucket_name, key):
    load_dotenv()
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    region_name = os.getenv("AWS_REGION", "us-east-1")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name
    )

    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def load_model(path: str = MODEL_PATH) -> Any:
    """Carrega um modelo treinado a partir de um arquivo .pkl."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    print(f"Carregando modelo de: {path}")
    return joblib.load(path)

def load_dataset(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset processado a partir de um arquivo .csv."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")
    print(f"Carregando dataset de: {path}")
    return pd.read_csv(path)

def save_model(model: Any, path: str = MODEL_PATH):
    """Salva um objeto (modelo, encoder, etc.) em um arquivo .pkl."""
    print(f"Salvando modelo em: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def prepare_data_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o DataFrame para predição, removendo colunas não numéricas
    e garantindo que o output seja puramente numérico.
    """
    print("Limpando dados para predição...")
    df_numeric = df.select_dtypes(include=['number', 'bool'])
    original_cols = set(df.columns)
    numeric_cols = set(df_numeric.columns)
    discarded_cols = original_cols - numeric_cols
    
    if discarded_cols:
        print(f"Colunas não-numéricas descartadas: {len(discarded_cols)}")
    return df_numeric
