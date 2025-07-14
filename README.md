# Previsão de Irradiância Solar e Geração Distribuída

Este projeto implementa um fluxo de trabalho completo para a previsão de irradiância solar e o cálculo da geração de energia para usinas solares distribuídas, utilizando dados climáticos e de empreendimentos.

## Estrutura do Projeto

A estrutura de pastas e arquivos é organizada da seguinte forma:

TCC-Previsao-Irradiancia-Geracao-MMGD/
├── data/
│   ├── DADOS PREC SUBSUL-20250714T010521Z-1-001.zip
│   ├── DADOS TEMP SUBSUL-20250714T010441Z-1-001.zip
│   ├── dados_aneeltec.csv
│   ├── dados_aneeltec.xlsx
│   ├── irrad_sub_sul-20250714T010351Z-1-001.zip
│   └── municipios.csv
├── .gitignore
├── LICENSE
├── data_processing.py
├── generation_calculations.py
├── model_training.py
└── README.md

## Descrição do Projeto

O projeto é dividido em três etapas principais, cada uma com seu script dedicado:

1.  **Processamento de Dados (`data_processing.py`):** Este script é responsável por carregar os dados históricos de irradiância, temperatura e precipitação de fontes como o banco de dados NASA POWER. Ele realiza a limpeza de dados (incluindo a correção de valores ausentes usando KMeans para dados de irradiância) e agrupa as informações por coordenadas geográficas e data, resultando em um dataset unificado e pronto para o treinamento do modelo.

2.  **Treinamento do Modelo (`model_training.py`):** Utiliza o dataset processado para construir e treinar um modelo de Machine Learning (XGBoost) para prever a irradiância solar. O processo inclui a otimização de hiperparâmetros com `RandomizedSearchCV` para encontrar a melhor configuração e a avaliação de desempenho usando métricas como Mean Absolute Error (MAE), R-quadrado (R²), Root Mean Squared Error (RMSE) e Mean Absolute Percentage Error (MAPE). Após o treinamento, o script também gera previsões de irradiância para um período futuro especificado.

3.  **Cálculos de Geração (`generation_calculations.py`):** Este script pega as previsões de irradiância geradas e as associa a uma base de dados de empreendimentos de geração solar (como dados da ANEEL). A associação é feita com base na proximidade geográfica das usinas aos pontos de previsão. Em seguida, calcula a geração de energia esperada para cada usina, aplicando fatores como a taxa de degradação anual dos painéis e a eficiência de conversão do sistema.

## Configuração e Instalação

Para configurar e executar o projeto no seu ambiente local, siga os passos abaixo:

1.  **Clone o Repositório:**
    Abra seu terminal (ou Git Bash no Windows) e navegue até o diretório onde deseja armazenar o projeto. Use o comando `git clone` seguido da URL do seu repositório GitHub.

    ```bash
    git clone [https://github.com/laslana/TCC-Previsao-Irradiancia-Geracao-MMGD.git](https://github.com/laslana/TCC-Previsao-Irradiancia-Geracao-MMGD.git)
    cd TCC-Previsao-Irradiancia-Geracao-MMGD
    ```
    (A URL acima foi baseada na sua imagem do GitHub e é um exemplo. Certifique-se de usar a URL exata do seu repositório se ela for diferente.)

2.  **Crie o Ambiente Virtual (Opcional, mas Recomendado):**
    É uma boa prática criar um ambiente virtual para isolar as dependências do seu projeto.
    ```bash
    python -m venv venv
    ```
    Ative o ambiente virtual:
    * No Linux/macOS: `source venv/bin/activate`
    * No Windows: `venv\Scripts\activate`

3.  **Instale as Dependências:**
    Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:
    ```bash
    pip install pandas numpy scikit-learn xgboost scipy
    ```

4.  **Preparação dos Dados (Manual):**
    * Os arquivos de dados da NASA POWER (`DADOS PREC SUBSUL-*.zip`, `DADOS TEMP SUBSUL-*.zip`, `irrad_sub_sul-*.zip`) estão no formato ZIP na pasta `data/`.
    * **Você precisará descompactar esses arquivos manualmente na sua máquina local.**
    * **Crie as pastas necessárias:** Se seus scripts esperam que os arquivos CSV descompactados estejam em subpastas como `data/irrad_sub_sul/`, `data/DADOS_TEMP_SUBSUL/` e `data/DADOS_PREC_SUBSUL/`, você deve criar essas pastas e colocar os CSVs descompactados nelas.
    * Verifique também se o `dados_aneeltec.xlsx` precisa ser convertido para `.csv` para ser lido pelos scripts, e qual o caminho correto para `municipios.csv`.

## Uso

Execute os scripts na ordem para garantir que os dados sejam processados corretamente e que as previsões e cálculos dependam dos passos anteriores. Certifique-se de que seu ambiente virtual esteja ativado e que os dados estejam preparados conforme a seção anterior.

1.  **Processar os Dados:**
    ```bash
    python data_processing.py
    ```
    Este script espera encontrar os dados brutos nos caminhos especificados internamente. Após a execução, ele salvará o arquivo `df_nasa_grouped.csv` na pasta `data/processed_data/` (se essa pasta for criada pelo script, ou você precisará criá-la).

2.  **Treinar o Modelo e Gerar Previsões:**
    ```bash
    python model_training.py
    ```
    Este script carregará `df_nasa_grouped.csv` (da `data/processed_data/`), treinará o modelo XGBoost, realizará a busca de hiperparâmetros e gerará as previsões futuras, salvando-as em `df_future_predictions.csv` na pasta `data/processed_data/`. Você pode ajustar as datas de previsão `START_DATE` e `END_DATE` dentro deste arquivo (`model_training.py`).

3.  **Calcular a Geração de Energia:**
    ```bash
    python generation_calculations.py
    ```
    Este script carregará os dados dos empreendimentos (ex: `dados_aneeltec.csv`) e `df_future_predictions.csv`, associará as previsões às usinas e calculará a geração de energia, salvando o resultado final em `df_final_generation.csv` na pasta `data/processed_data/`.

## Dados

* **Dados de Irradiância (NASA POWER):** Arquivos ZIP contendo os dados CSV diários de irradiância solar global de superfície (ALLSKY_SFC_SW_DWN). Precisam ser descompactados para uso.
* **Dados de Temperatura e Precipitação (NASA POWER):** Arquivos ZIP contendo os dados CSV diários com temperatura (T2M) e precipitação (PRECTOTCORR). Precisam ser descompactados para uso.
* **Dados dos Empreendimentos (ANEEL Tec):** Arquivos CSV/XLSX contendo informações sobre as usinas de geração distribuída, incluindo coordenadas de latitude e longitude, potência instalada e data de conexão.

## Contato

Para dúvidas ou sugestões, sinta-se à vontade para entrar em contato:
Laís Lana de Pinho
[laislanap@gmail.com ou https://www.linkedin.com/in/lais-pinho/ ]

## Licença

Este projeto está licenciado sob a [Licença MIT](https://opensource.org/licenses/MIT).
