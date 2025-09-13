import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a engenharia de features no DataFrame, usando Target Encoding
    para categóricas e criando features de interação.
    """
    df = df.copy()
    print("Iniciando engenharia de features avançada (sem NLP)...")

    # --- Features de Interação (Match Vaga vs. Candidato) ---
    print("Criando features de interação...")
    
    # Match de Nível Profissional (ex: Pleno vs Sênior)
    if 'perfil_vaga_nivel_profissional' in df.columns and 'informacoes_profissionais_nivel_profissional' in df.columns:
        # Mapear para valores numéricos para comparação
        level_map = {'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Especialista': 4}
        vaga_level = df['perfil_vaga_nivel_profissional'].map(level_map)
        cand_level = df['informacoes_profissionais_nivel_profissional'].map(level_map)
        # Feature: 1 se o nível do candidato for igual ou superior ao da vaga, 0 caso contrário
        df['match_nivel_profissional'] = (cand_level >= vaga_level).astype(int)

    # Match de Localização (Cidade)
    if 'perfil_vaga_cidade' in df.columns and 'informacoes_pessoais_local' in df.columns:
        # O 'local' do candidato pode ser "Cidade, Estado", então extraímos a cidade
        cand_cidade = df['informacoes_pessoais_local'].str.split(',').str[0]
        df['match_cidade'] = (df['perfil_vaga_cidade'].str.lower() == cand_cidade.str.lower()).astype(int)

    # --- Suas features existentes (ótimo mantê-las) ---
    if 'cv_experience_years' in df.columns:
        df['experience_bin'] = pd.cut(df['cv_experience_years'], bins=[0, 1, 3, 5, 10, 30], labels=False, right=False)

    if 'cv_total_skills' in df.columns:
        df['skills_bin'] = pd.cut(df['cv_total_skills'], bins=[0, 3, 6, 10, 20, 50], labels=False, right=False)

    if 'cv_word_count' in df.columns and 'cv_total_skills' in df.columns:
        df['cv_complexity'] = df['cv_word_count'] * df['cv_total_skills']

    # ... (o resto das suas features) ...
    today = pd.to_datetime('today')
    if 'informacoes_pessoais_data_nascimento' in df.columns:
        df['idade'] = today.year - pd.to_datetime(df['informacoes_pessoais_data_nascimento'], errors='coerce').dt.year

    # --- Target Encoding para Categóricas de Alta Cardinalidade ---
    print("Aplicando Target Encoding...")
    
    # Identificar colunas categóricas com muitos valores únicos
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = []
    for col in categorical_cols:
        if df[col].nunique() > 10 and col != 'target': # Ajuste o threshold '10' se necessário
            high_cardinality_cols.append(col)
            
    if high_cardinality_cols and 'target' in df.columns:
        print(f"Colunas para Target Encoding: {high_cardinality_cols}")
        # O TargetEncoder precisa do 'y' (alvo) para o cálculo
        # É importante fazer isso ANTES do split de treino/teste para evitar data leakage
        # No pipeline, garantimos que o 'target' está presente
        encoder = TargetEncoder(cols=high_cardinality_cols)
        
        # Separar X e y temporariamente para o encoder
        y = df['target']
        X = df.drop(columns=['target'])
        
        # Aplicar o encoding
        X_encoded = encoder.fit_transform(X, y)
        
        # Juntar novamente
        df = pd.concat([X_encoded, y], axis=1)
    
    print("Engenharia de features concluída.")
    return df

def pipeline_feature_engineering():
    print('--- Iniciando Pipeline de Engenharia de Features ---')
    # Ler do 'preprocessed_data.csv'
    df = pd.read_csv("src/data/processed/preprocessed_data.csv", parse_dates=True)
    
    # O Target Encoder precisa da coluna 'target', então garantimos que ela está lá
    if 'target' not in df.columns:
        raise ValueError("A coluna 'target' é necessária para o Target Encoding e não foi encontrada.")
        
    df_featured = feature_engineering(df)
    df_featured.to_csv("src/data/processed/feature_engineered_data.csv", index=False, encoding="utf-8")
    print("\nCSV com novas features salvo em src/data/processed/feature_engineered_data.csv")
    print('--- Pipeline de Engenharia de Features Concluído ---')

if __name__ == "__main__":
    pipeline_feature_engineering()
