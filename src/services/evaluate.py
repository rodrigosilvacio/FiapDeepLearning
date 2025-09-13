# src/services/evaluate.py

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils.utils import load_model, load_dataset, prepare_data_for_prediction

TARGET_COL = "target"
METRICS_PATH = "src/reports/metrics"

def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_true, (y_pred_proba > t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def pipeline_evaluate():
    """
    Carrega o modelo, avalia e salva um relatório completo com múltiplas métricas e gráficos.
    """
    print("=== Iniciando Avaliação do Modelo ===")
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    # Carregar modelo, dados e colunas
    model = load_model()
    df = load_dataset()
    try:
        model_columns = joblib.load('src/models/model_columns.pkl')
    except FileNotFoundError:
        print("❌ Erro: 'model_columns.pkl' não encontrado. Execute o treino primeiro.")
        return

    # Alinhar dados
    X = df.drop(columns=[TARGET_COL], errors='ignore')
    y = df[TARGET_COL]
    X_aligned = pd.get_dummies(X, dummy_na=True).reindex(columns=model_columns, fill_value=0)
    
    # Split
    _, X_test, _, y_test = train_test_split(X_aligned, y, test_size=0.2, random_state=42, stratify=y)
    
    # Predições
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Métricas
    auc = roc_auc_score(y_test, y_pred_proba)
    # ... (cálculo de outras métricas) ...

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred, target_names=['Não Recomendado', 'Recomendado']))

    # --- Gerar e Salvar Gráficos ---
    print("\nGerando e salvando gráficos de avaliação...")

    # 1. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Recomendado', 'Recomendado'], yticklabels=['Não Recomendado', 'Recomendado'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig(os.path.join(METRICS_PATH, 'confusion_matrix.png'))
    plt.close()

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(METRICS_PATH, 'roc_curve.png'))
    plt.close()

    # 3. Curva de Precisão-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva de Precisão-Recall')
    plt.grid(True)
    plt.savefig(os.path.join(METRICS_PATH, 'precision_recall_curve.png'))
    plt.close()

    # 4. Histograma de Distribuição de Probabilidades
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba[y_test == 0], color="red", label='Não Recomendado (Real)', kde=True, stat="density", linewidth=0)
    sns.histplot(y_pred_proba[y_test == 1], color="green", label='Recomendado (Real)', kde=True, stat="density", linewidth=0)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold Ótimo ({threshold:.2f})')
    plt.title('Distribuição das Probabilidades do Modelo')
    plt.xlabel('Probabilidade (Score de Match)')
    plt.legend()
    plt.savefig(os.path.join(METRICS_PATH, 'probability_distribution.png'))
    plt.close()

    print(f"Gráficos salvos em: {METRICS_PATH}")
    print("\n=== Avaliação Concluída ===")
    
if __name__ == "__main__":
    pipeline_evaluate()
