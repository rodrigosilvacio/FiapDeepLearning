import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path
from utils.utils import read_csv_s3

# Configuração da página
st.set_page_config(
    page_title="FIAP Sistema de Matching Vaga-Candidato",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminhos dos arquivos (ajustados para a estrutura do projeto)
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
MODEL_COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"

# Cache para carregar recursos pesados apenas uma vez
@st.cache_resource
def load_resources():
    load_dotenv()
    """Carrega o modelo, colunas e dados com cache para melhor performance."""
    try:
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        # Carregar modelo
        model = joblib.load(MODEL_PATH)
        
        # Carregar colunas do modelo
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
        
        # Carregar dados com features (para predição)
        df_featured = read_csv_s3(bucket_name, "feature_engineered_data.csv")
        
        # Carregar dados processados (para exibição de informações legíveis)
        df_processed = read_csv_s3(bucket_name, "preprocessed_data.csv")
        
        return model, model_columns, df_featured, df_processed
    
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        st.error("Certifique-se de que o modelo foi treinado e os dados estão na pasta correta.")
        return None, None, None, None

def prepare_data_for_prediction(df, model_columns):
    """Prepara os dados para predição, alinhando com as colunas do modelo."""
    # Aplicar one-hot encoding
    df_processed = pd.get_dummies(df, dummy_na=True)
    
    # Alinhar com as colunas do modelo
    df_aligned = df_processed.reindex(columns=model_columns, fill_value=0)
    
    return df_aligned

def main():
    # Título principal
    st.title("🤖 Sistema Inteligente de Matching Vaga-Candidato")
    st.markdown("""
    Esta aplicação utiliza um modelo de Machine Learning para prever a compatibilidade 
    entre candidatos e vagas, ajudando recrutadores a otimizarem seu tempo e encontrarem 
    os melhores matches.
    """)
    
    # Carregar recursos
    with st.spinner("Carregando modelo e dados... Por favor, aguarde."):
        model, model_columns, df_featured, df_processed = load_resources()
    
    if model is None:
        st.stop()
    
    # Sidebar com informações do modelo
    st.sidebar.header("📊 Informações do Modelo")
    st.sidebar.metric("Features Utilizadas", len(model_columns))
    st.sidebar.metric("Total de Registros", len(df_featured))
    
    #Comentando tab2
    # Separar em abas para melhor organização
    tab1, tab3 = st.tabs(["🎯 Análise por Vaga", "ℹ️ Sobre o Modelo"])
    
    with tab1:
        st.header("Análise de Candidatos por Vaga")
        
        # Seletor de vaga usando dados processados
        if 'titulo_vaga' in df_processed.columns:
            job_titles = df_processed['titulo_vaga'].dropna().unique()
            selected_job = st.selectbox(
                "Selecione uma Vaga:",
                options=job_titles,
                help="Escolha uma vaga para ver os candidatos mais compatíveis"
            )
            
            if selected_job:
                # Encontrar o job_id correspondente
                job_data = df_processed[df_processed['titulo_vaga'] == selected_job]
                
                if not job_data.empty:
                    job_id = job_data['job_id'].iloc[0]
                    
                    # Filtrar candidatos para esta vaga no dataset com features
                    df_job = df_featured[df_featured['job_id'] == job_id].copy()
                    
                    if df_job.empty:
                        st.warning("Nenhum candidato encontrado para esta vaga no dataset processado.")
                    else:
                        # Preparar dados para predição
                        X_job = df_job.drop(columns=['target'], errors='ignore')
                        X_job_prepared = prepare_data_for_prediction(X_job, model_columns)
                        
                        # Fazer predições
                        probabilities = model.predict_proba(X_job_prepared)[:, 1]
                        df_job['score_match'] = probabilities
                        
                        # Juntar com dados processados para exibição
                        display_data = pd.merge(
                            df_job[['applicant_id', 'score_match']],
                            df_processed[['applicant_id', 'nome', 'cv_experience_years', 'cv_total_skills']].drop_duplicates(subset=['applicant_id']),
                            on='applicant_id',
                            how='left'
                        )
                        
                        # Controles de exibição
                        col1, col2 = st.columns(2)
                        with col1:
                            top_n = st.slider("Número de candidatos a exibir:", 1, min(20, len(display_data)), 5)
                        with col2:
                            min_score = st.slider("Score mínimo:", 0.0, 1.0, 0.0, 0.05)
                        
                        # Filtrar e ordenar
                        filtered_data = display_data[display_data['score_match'] >= min_score]
                        top_candidates = filtered_data.sort_values(by='score_match', ascending=False).head(top_n)
                        
                        if top_candidates.empty:
                            st.warning("Nenhum candidato encontrado com o score mínimo especificado.")
                        else:
                            st.subheader(f"🏆 Top {len(top_candidates)} Candidatos para: {selected_job}")
                            
                            # Exibir tabela com formatação
                            st.dataframe(
                                top_candidates,
                                column_config={
                                    "applicant_id": st.column_config.TextColumn("ID Candidato", width="small"),
                                    "nome": st.column_config.TextColumn("Nome", width="medium"),
                                    "score_match": st.column_config.ProgressColumn(
                                        "Score de Match",
                                        format="%.3f",
                                        min_value=0,
                                        max_value=1,
                                        width="medium"
                                    ),
                                    "cv_experience_years": st.column_config.NumberColumn("Anos de Exp.", width="small"),
                                    "cv_total_skills": st.column_config.NumberColumn("Nº de Skills", width="small")
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Estatísticas rápidas
                            st.subheader("📊 Estatísticas dos Candidatos Selecionados")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Score Médio", f"{top_candidates['score_match'].mean():.3f}")
                            with col2:
                                st.metric("Score Máximo", f"{top_candidates['score_match'].max():.3f}")
                            with col3:
                                exp_mean = top_candidates['cv_experience_years'].mean()
                                st.metric("Experiência Média", f"{exp_mean:.1f} anos" if not pd.isna(exp_mean) else "N/A")
                            with col4:
                                skills_mean = top_candidates['cv_total_skills'].mean()
                                st.metric("Skills Médias", f"{skills_mean:.0f}" if not pd.isna(skills_mean) else "N/A")
                else:
                    st.error("Erro ao encontrar dados para a vaga selecionada.")
        else:
            st.error("Coluna 'titulo_vaga' não encontrada nos dados processados.")
    
    # with tab2:
    #     st.header("📈 Estatísticas Gerais do Dataset")
        
    #     # Métricas gerais
    #     col1, col2, col3, col4 = st.columns(4)
        
    #     with col1:
    #         st.metric("Total de Vagas", df_processed['job_id'].nunique() if 'job_id' in df_processed.columns else "N/A")
    #     with col2:
    #         st.metric("Total de Candidatos", df_processed['applicant_id'].nunique() if 'applicant_id' in df_processed.columns else "N/A")
    #     with col3:
    #         st.metric("Total de Candidaturas", len(df_processed))
    #     with col4:
    #         if 'target' in df_featured.columns:
    #             match_rate = df_featured['target'].mean() * 100
    #             st.metric("Taxa de Match", f"{match_rate:.1f}%")
    #         else:
    #             st.metric("Taxa de Match", "N/A")
        
    #     # Gráficos simples
    #     if 'cv_experience_years' in df_processed.columns:
    #         st.subheader("Distribuição de Experiência dos Candidatos")
    #         experience_data = df_processed['cv_experience_years'].dropna()
    #         if not experience_data.empty:
    #             st.bar_chart(experience_data.value_counts().sort_index())
    #         else:
    #             st.info("Dados de experiência não disponíveis.")
        
    #     # Distribuição de skills
    #     if 'cv_total_skills' in df_processed.columns:
    #         st.subheader("Distribuição de Total de Skills")
    #         skills_data = df_processed['cv_total_skills'].dropna()
    #         if not skills_data.empty:
    #             st.bar_chart(skills_data.value_counts().sort_index())
    #         else:
    #             st.info("Dados de skills não disponíveis.")
    
    with tab3:
        st.header("ℹ️ Sobre o Modelo")
        
        st.markdown("""
        ### Como Funciona o Sistema
        
        Este sistema utiliza um modelo de Machine Learning (LightGBM) treinado para prever a compatibilidade 
        entre candidatos e vagas com base em diversas características:
        
        - **Experiência profissional** do candidato
        - **Habilidades técnicas** listadas no currículo
        - **Localização** geográfica
        - **Nível profissional** (Júnior, Pleno, Sênior)
        - **Histórico de sucesso** de recrutadores e cidades (via Target Encoding)
        
        ### Métricas de Performance
        
        O modelo foi avaliado e apresenta as seguintes métricas:
        - **AUC**: ~0.87 (Excelente capacidade de discriminação)
        - **F1-Score**: ~0.68 (Boa precisão e recall balanceados)
        - **Acurácia**: ~78% (Alta taxa de acertos)
        
        ### Como Interpretar os Scores
        
        - **Score > 0.8**: Candidato altamente compatível
        - **Score 0.6 - 0.8**: Candidato com boa compatibilidade
        - **Score 0.4 - 0.6**: Candidato com compatibilidade moderada
        - **Score < 0.4**: Candidato com baixa compatibilidade
        
        ### Limitações
        
        - O modelo é baseado em dados históricos e pode ter vieses
        - Não considera aspectos subjetivos como fit cultural
        - Recomenda-se sempre uma análise manual final do recrutador
        """)

if __name__ == "__main__":
    main()