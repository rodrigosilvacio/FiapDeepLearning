import json
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}")
        return {}

def flatten_jobs(job_json: dict, key_prefix: str) -> pd.DataFrame:
    """Flatten JSON aninhado em DataFrame"""
    rows = []
    for job_id, job_data in job_json.items():
        flat = {"job_id": job_id}
        if key_prefix == "vagas":
            for section in ["informacoes_basicas", "perfil_vaga", "beneficios"]:
                for k, v in job_data.get(section, {}).items():
                    flat[f"{section}_{k}"] = v
        elif key_prefix == "prospects":
            for p in job_data.get("prospects", []):
                row = p.copy()
                row['job_id'] = job_id
                row['titulo_vaga'] = job_data.get("titulo", "")
                row['modalidade_vaga'] = job_data.get("modalidade", "")
                rows.append(row)
            continue
        rows.append(flat)
    return pd.DataFrame(rows)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(["", "0000-00-00"], np.nan)
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna("Não informado")
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(0)
    return df

def extract_cv_features(df: pd.DataFrame, cv_col="cv_pt") -> pd.DataFrame:
    df = df.copy()
    if cv_col not in df.columns:
        return df
    df["cv_word_count"] = df[cv_col].apply(lambda x: len(str(x).split()))
    df["cv_char_count"] = df[cv_col].apply(lambda x: len(str(x)))
    df["cv_has_content"] = df[cv_col].apply(lambda x: 1 if len(str(x).strip()) > 10 else 0)

    # Experiência
    df["cv_experience_years"] = df[cv_col].apply(
        lambda x: min(sum(int(y[0]) for y in re.findall(r'(\d+)\s*(anos|ano|years|year|yr|y)', str(x).lower()) if int(y[0])<50), 30)
    )
    
    # Inglês
    levels = {"basico":1, "básico":1,"iniciante":1,"basic":1,"intermediario": 2, "intermediário":2,"intermedio":2,"intermediate":2,
              "avançado":3,"advanced":3,"fluente":4,"fluent":4,"nativo":4,"native":4}
    df["cv_english_level"] = df[cv_col].apply(lambda x: max([v for k,v in levels.items() if k in str(x).lower()]+[0]))
    
    # Skills simples
    skills = ["python","java","sql","javascript","html","css","aws","azure","cloud","docker","kubernetes",
              "machine learning","ai","data science","big data","excel","power bi","tableau","sql server",
              "mysql","nosql","mongodb","postgresql","oracle","linux","windows","git","jenkins","ci/cd","agile","scrum"]
    for skill in skills:
        df[f"cv_skill_{skill}"] = df[cv_col].apply(lambda x: str(x).lower().count(skill))
    df["cv_total_skills"] = df[[f"cv_skill_{s}" for s in skills]].sum(axis=1)
    return df

def encode_and_normalize(df: pd.DataFrame, categorical_cols=[], numerical_cols=[]):
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    scaler = MinMaxScaler()
    valid_numerical = [c for c in numerical_cols if c in df.columns]
    if valid_numerical:
        df[valid_numerical] = scaler.fit_transform(df[valid_numerical])
    return df, encoders, scaler

# ------------------------------
# Função principal
# ------------------------------

def pipeline_preprocessing(applicants_path, prospects_path, vagas_path):
    # Carregar JSONs
    print("Carregando JSONs...")
    applicants = load_json(applicants_path)
    prospects = load_json(prospects_path)
    vagas = load_json(vagas_path)
    
    # Transformar em DataFrame
    print("Transformando em DataFrame...")
    applicants_df = pd.DataFrame.from_dict(applicants, orient='index').reset_index(drop=True)
    prospects_df = flatten_jobs(prospects, "prospects")
    vagas_df = flatten_jobs(vagas, "vagas")
    
    # Limpeza
    print("Limpeza de dados...")
    applicants_df = clean_df(applicants_df)
    prospects_df = clean_df(prospects_df)
    vagas_df = clean_df(vagas_df)
    
    # Features de CV
    print("Extração de features de CV...")
    applicants_df = extract_cv_features(applicants_df)
    
    # Encoding + Normalização
    print("Encoding + Normalização...")
    categorical_cols = ["informacoes_pessoais_sexo",
                        "formacao_e_idiomas_nivel_ingles",
                        "formacao_e_idiomas_nivel_espanhol", 
                        "informacoes_profissionais_area_atuacao"]
    numerical_cols = ["cv_word_count","cv_char_count","cv_experience_years","cv_total_skills"]
    applicants_df, encoders, scaler = encode_and_normalize(applicants_df, categorical_cols, numerical_cols)
    
    # Merge final
    print("Merge final...")
    applicants_df['applicant_id'] = applicants_df.index.astype(str)
    df = prospects_df.merge(applicants_df, left_on="codigo", right_on="applicant_id", how="left")
    df = df.merge(vagas_df, on="job_id", how="left")
    
    # Target simplificada
    print("Target simplificado...")
    df['target'] = df['situacao_candidado'].apply(lambda x: 1 if 'encaminhado' in str(x).lower() else 0)

    print(f"Pré-processamento concluído: {len(df)} linhas, {len(df.columns)} colunas")
    return df, encoders, scaler

# ------------------------------
# Execução
# ------------------------------
if __name__ == "__main__":
    print('Iniciando pre-processamento...')
    df, encoders, scaler = pipeline_preprocessing("src/data/raw/applicants.json",
                                "src/data/raw/prospects.json",
                                "src/data/raw/vagas.json")
    df.to_csv("src/data/processed/preprocessed_data.csv", index=False, encoding="utf-8")
    print("CSV salvo em src/data/processed/preprocessed_data.csv")