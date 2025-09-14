# Sistema Inteligente de Matching Vaga-Candidato

## 📄 Descrição do Projeto

Este projeto apresenta um sistema avançado de Machine Learning projetado para otimizar o processo de recrutamento e seleção de talentos. O objetivo principal é automatizar e aprimorar a compatibilidade entre candidatos e vagas disponíveis, utilizando algoritmos de aprendizado de máquina para identificar os "melhores matches" com alta precisão. Em um cenário onde o volume de candidaturas pode ser esmagador, esta ferramenta visa reduzir significativamente o tempo gasto por recrutadores na triagem manual, permitindo que se concentrem em interações mais estratégicas com candidatos qualificados.

O sistema processa dados de currículos de candidatos e descrições de vagas, transformando informações textuais e estruturadas em features numéricas que podem ser compreendidas por modelos de Machine Learning. A arquitetura do projeto é modular, garantindo escalabilidade, manutenibilidade e reprodutibilidade de todo o pipeline de dados e modelagem. Desde o pré-processamento de dados brutos (JSONs) até a engenharia de features complexas, treinamento de modelos preditivos e avaliação de performance, cada etapa é cuidadosamente orquestrada para entregar resultados robustos e acionáveis.

O resultado final é uma aplicação interativa construída com Streamlit, que oferece uma interface amigável para recrutadores. Através dela, é possível selecionar uma vaga específica e visualizar uma lista ranqueada de candidatos com base em seu score de compatibilidade, permitindo uma tomada de decisão mais rápida e informada. Este sistema não apenas acelera o processo de matching, mas também busca introduzir uma camada de objetividade e consistência na avaliação de candidatos, complementando a expertise humana com o poder da inteligência artificial.

## 📄 Sobre o modelo utilizado

Queremos apoiar empregadores e candidatos na previsão da compatibilidade entre vagas.

Experiência profissional do candidato
- Habilidades técnicas listadas no currículo
- Localização geográfica
- Nível profissional (Júnior, Pleno, Sênior)
- Histórico de sucesso de recrutadores e cidades (via Target Encoding)
- Métricas de Performance

O modelo foi avaliado e apresenta as seguintes métricas:
- AUC: ~0.87 (Excelente capacidade de discriminação)
- F1-Score: ~0.68 (Boa precisão e recall balanceados)
- Acurácia: ~78% (Alta taxa de acertos)

Como Interpretar os Scores
- Score > 0.8: Candidato altamente compatível
- Score 0.6 - 0.8: Candidato com boa compatibilidade
- Score 0.4 - 0.6: Candidato com compatibilidade moderada
- Score < 0.4: Candidato com baixa compatibilidade

Limitações
O modelo é baseado em dados históricos e pode ter vieses
Não considera aspectos subjetivos como fit cultural
Recomenda-se sempre uma análise manual final do recrutador

## 🛠️ Stack Utilizada

Este projeto foi desenvolvido utilizando uma combinação de tecnologias e bibliotecas Python, focando em robustez, eficiência e facilidade de uso para o desenvolvimento de soluções de Machine Learning e aplicações web interativas. A escolha da stack reflete as melhores práticas em MLOps (Machine Learning Operations), garantindo um pipeline de dados e modelo bem estruturado e escalável.

### Linguagem de Programação

*   **Python**: A linguagem principal utilizada em todo o projeto, desde o pré-processamento de dados até o desenvolvimento do modelo e da interface web. Sua vasta gama de bibliotecas e a comunidade ativa a tornam ideal para projetos de Machine Learning.

### Bibliotecas Principais

*   **Pandas**: Essencial para manipulação e análise de dados, utilizada extensivamente nas etapas de pré-processamento e engenharia de features. Facilita a leitura, limpeza, transformação e agregação de grandes volumes de dados.
*   **NumPy**: Fundamental para operações numéricas de alta performance, servindo como base para muitas outras bibliotecas científicas em Python.
*   **Scikit-learn**: Uma biblioteca abrangente para Machine Learning, utilizada para tarefas como divisão de dados (treino/teste), pré-processamento (e.g., `MinMaxScaler`, `LabelEncoder`), e avaliação de modelos (e.g., `roc_auc_score`, `f1_score`, `accuracy_score`, `classification_report`, `confusion_matrix`).
*   **LightGBM**: O algoritmo de Machine Learning escolhido para o treinamento do modelo preditivo. Conhecido por sua alta performance, velocidade e eficiência no manuseio de grandes datasets, é uma excelente escolha para problemas de classificação como o matching de vagas.
*   **Optuna**: Utilizado para a otimização de hiperparâmetros do modelo LightGBM. Optuna é uma biblioteca de otimização automática de hiperparâmetros que permite encontrar as melhores configurações para o modelo de forma eficiente, melhorando significativamente a performance.
*   **Streamlit**: Framework utilizado para construir a interface de usuário interativa do sistema de matching. Permite transformar scripts Python em aplicações web ricas e dinâmicas com poucas linhas de código, ideal para prototipagem e demonstrações rápidas.
*   **Boto3**: A SDK (Software Development Kit) da AWS para Python, utilizada para interagir com serviços da Amazon Web Services, especificamente para carregar e salvar arquivos CSV no Amazon S3, garantindo o armazenamento escalável e seguro dos dados processados.
*   **python-dotenv**: Utilizada para carregar variáveis de ambiente de um arquivo `.env`, facilitando o gerenciamento de credenciais e configurações sensíveis (como chaves de acesso da AWS) sem expô-las diretamente no código-fonte.
*   **Joblib**: Biblioteca para serialização e desserialização de objetos Python, utilizada para salvar e carregar o modelo treinado (`model.pkl`) e as colunas do modelo (`model_columns.pkl`), permitindo que o modelo seja persistido e reutilizado sem a necessidade de retreinamento.
*   **Matplotlib** e **Seaborn**: Bibliotecas para criação de gráficos e visualizações de dados. Embora a versão final do Streamlit possa ter optado por gráficos mais simples do próprio Streamlit, estas bibliotecas são cruciais para a Análise Exploratória de Dados (EDA) e a visualização de métricas de avaliação do modelo em notebooks de desenvolvimento.

### Gerenciamento de Dependências

*   **`requirements.txt`**: Um arquivo que lista todas as dependências do projeto, permitindo que o ambiente de desenvolvimento seja facilmente replicado em qualquer máquina.

### Estrutura de Projeto

O projeto segue uma estrutura modular, com pastas dedicadas para código-fonte (`src`), dados (`data`), modelos (`models`), notebooks (`notebooks`), e a aplicação Streamlit (`app`), promovendo a organização e a manutenibilidade do código.



## 🚀 Como Rodar o Aplicativo Localmente

Para executar a aplicação Streamlit em sua máquina local, siga os passos abaixo. Certifique-se de que você tem o Python instalado (versão 3.8+ é recomendada).

### 1. Clonar o Repositório

Primeiro, clone este repositório para sua máquina local usando Git:

```bash
git clone https://github.com/usuario/BOSS_IA_Recrutamento.git
cd seu-repositorio
```

Substitua `https://github.com/seu-usuario/seu-repositorio.git` pelo URL real do seu repositório no GitHub.

### 2. Configurar o Ambiente Virtual

É altamente recomendável usar um ambiente virtual para gerenciar as dependências do projeto. Isso evita conflitos com outras instalações Python em sua máquina.

```bash
python -m venv venv
```

### 3. Ativar o Ambiente Virtual

*   **No Windows:**
    ```bash
    .\venv\Scripts\activate
    ```

*   **No macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Configurar Variáveis de Ambiente (AWS S3)

O aplicativo Streamlit lê os dados processados e com features do Amazon S3. Você precisará configurar suas credenciais AWS e o nome do bucket. Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```
AWS_ACCESS_KEY_ID=SUA_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=SUA_SECRET_ACCESS_KEY
AWS_REGION_NAME=sua-regiao-aws # Ex: us-east-1
AWS_BUCKET_NAME=seu-nome-do-bucket-s3
```

Substitua os valores pelos seus próprios. **Nunca compartilhe este arquivo `.env` publicamente (ele já está no `.gitignore` para sua segurança).**

### 6. Treinar o Modelo (Primeira Vez ou Retreinamento)

Antes de rodar o aplicativo, você precisa garantir que o modelo foi treinado e que os arquivos `model.pkl` e `model_columns.pkl` (e os CSVs processados) existem. Se você ainda não treinou o modelo ou se deseja retreiná-lo com dados atualizados, execute o script `main.py`:

```bash
python -m src.main
```

Este script irá:
*   Pré-processar os dados brutos (JSONs).
*   Realizar a engenharia de features.
*   Treinar o modelo de Machine Learning.
*   Avaliar o desempenho do modelo.
*   Salvar o modelo treinado e as colunas utilizadas.

**Importante:** Este passo também garante que os CSVs `preprocessed_data.csv` e `feature_engineered_data.csv` sejam gerados e, no seu caso, enviados para o S3, de onde o Streamlit os lerá.

### 7. Rodar o Aplicativo Streamlit

Após o treinamento do modelo e a configuração das variáveis de ambiente, você pode iniciar o aplicativo Streamlit:

```bash
streamlit run src/app/app.py
```

O Streamlit abrirá automaticamente o aplicativo em seu navegador padrão (geralmente em `http://localhost:8501`). Agora você pode interagir com o sistema de matching, selecionar vagas e visualizar os candidatos recomendados.



## 🔄 Como Treinar o Modelo Novamente

O processo de retreinamento do modelo é integrado ao script principal do projeto (`src/main.py`). Sempre que você tiver novos dados ou desejar atualizar o modelo com as últimas informações, basta executar este script.

### Passos para Retreinar o Modelo:

1.  **Garanta que seus dados brutos estejam atualizados:** Certifique-se de que os arquivos JSON (`applicants.json`, `prospects.json`, `vagas.json`) na pasta `src/data/raw/` contenham as informações mais recentes que você deseja usar para o treinamento.

2.  **Execute o script principal:** Com seu ambiente virtual ativado (conforme os passos de instalação acima), navegue até a raiz do projeto e execute:

    ```bash
    python -m src.main
    ```

    Este comando irá:
    *   Carregar os dados brutos atualizados.
    *   Executar todo o pipeline de pré-processamento e engenharia de features.
    *   Treinar um novo modelo de Machine Learning do zero, utilizando os dados mais recentes.
    *   Avaliar o desempenho do novo modelo.
    *   **Salvar o novo modelo treinado** (`model.pkl`) e as colunas utilizadas (`model_columns.pkl`) na pasta `src/models/`, sobrescrevendo as versões anteriores.
    *   Os CSVs processados e com features (`preprocessed_data.csv`, `feature_engineered_data.csv`) também serão atualizados e, no seu caso, enviados para o S3.

Após a conclusão bem-sucedida do `src/main.py`, seu aplicativo Streamlit (quando reiniciado ou acessado) automaticamente carregará o modelo recém-treinado e os dados atualizados, refletindo as melhorias ou as novas informações incorporadas.

---
