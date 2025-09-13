import pandas as pd
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import warnings
import os
import optuna  

# Desabilitar logs detalhados do Optuna para manter a saída limpa
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# Configurações
DATA_PATH = "src/data/processed/feature_engineered_data.csv"
MODEL_PATH = "src/models/model.pkl"
TARGET_COL = "target"

def load_data():
    """Carrega os dados"""
    df = pd.read_csv(DATA_PATH)
    return df

def prepare_features(df):
    """Prepara features para treinamento"""
    cols_to_drop = [
        TARGET_COL, 'job_id', 'codigo', 'applicant_id', 'nome',
        'data_candidatura', 'ultima_atualizacao', 'comentario', 'recrutador',
        'titulo_vaga', 'cv_pt', 'cv_en', 'situacao_candidado',
        'informacoes_pessoais_nome', 'informacoes_pessoais_email', 
        'informacoes_pessoais_cpf', 'informacoes_pessoais_telefone_celular',
        'informacoes_basicas_titulo_vaga', 'informacoes_basicas_vaga_sap',
        'infos_basicas', 'informacoes_pessoais', 'informacoes_profissionais', 
        'formacao_e_idiomas', 'cargo_atual'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]

    # Tratar categóricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    cols_removed = []
    for col in categorical_cols.copy():
        try:
            sample_str = X[col].dropna().astype(str).iloc[0] if len(X[col].dropna()) > 0 else ""
            if (X[col].nunique() > 50 or len(sample_str) > 100 or sample_str.startswith('{') or sample_str.startswith('[')):
                X = X.drop(columns=[col])
                categorical_cols.remove(col)
                cols_removed.append(col)
        except:
            X = X.drop(columns=[col])
            categorical_cols.remove(col)
            cols_removed.append(col)

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=True)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    constant_cols = X.columns[X.nunique() <= 1].tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    X = X.fillna(X.median())
    return X, y

def cross_validate_model(X, y):
    """Validação cruzada rápida"""
    scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
    model = LGBMClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores

# 2. Nova função 'objective' para o Optuna
def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'n_jobs': -1
    }

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'auc')]) # Pruning para otimizar a busca

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    return auc

# 3. Função de treino modificada para usar o Optuna
def train_model(X, y):
    """
    Executa a otimização de hiperparâmetros com Optuna e treina o modelo final.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # --- Otimização com Optuna ---
    print("🚀 Iniciando otimização de hiperparâmetros com Optuna...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    # Aumente n_trials para uma busca mais exaustiva (ex: 100), mas 30 já é um bom começo.
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight), n_trials=30)
    
    best_params = study.best_params
    print("✅ Otimização concluída!")
    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Melhor AUC na busca: {study.best_value:.4f}")
    
    # --- Treinamento do modelo final com os melhores parâmetros ---
    final_params = best_params.copy()
    final_params.update({
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1
    })
    
    model = LGBMClassifier(**final_params)
    model.fit(X_train, y_train)
    model_columns = X_train.columns.tolist()
    joblib.dump(model_columns, 'src/models/model_columns.pkl')
    print(f"Lista de {len(model_columns)} colunas do modelo salva em src/models/model_columns.pkl")

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Encontrar threshold ótimo para F1 no conjunto de validação
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_val, (y_pred_proba > t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return model, best_threshold, y_val, y_pred_proba, best_params

def save_model(model):
    """Salva o modelo"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def pipeline_train():
    try:
        df = load_data() 
        X, y = prepare_features(df)

        if X.shape[0] < 1000:
            print("⚠️ Poucos dados para treinamento!")
        if X.shape[1] < 5:
            raise ValueError(f"Muito poucas features após limpeza: {X.shape[1]}")
        
        model, best_threshold, y_val, y_pred_proba, best_params = train_model(X, y)
        save_model(model)
        model_columns = X.columns.tolist()
        joblib.dump(model_columns, 'src/models/model_columns.pkl')
        print(f"✅ Lista de {len(model_columns)} colunas do modelo salva em src/models/model_columns.pkl")
        y_pred = (y_pred_proba > best_threshold).astype(int)
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)

        print("=" * 60)
        print("✅ TREINAMENTO OTIMIZADO CONCLUÍDO COM SUCESSO!")
        print(f"AUC (validação): {auc:.4f}")
        print(f"F1 ajustado: {f1_score(y_val, y_pred):.4f}")
        print(f"Acurácia: {accuracy:.1%}")
        print(f"Threshold ótimo: {best_threshold:.2f}")
        print(f"Features utilizadas: {X.shape[1]}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    pipeline_train()
