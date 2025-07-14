# src/generation_calculations.py

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os

def associate_irradiance_to_plants(dados_empreend, df_future):
    """
    Associa as previsões de irradiância de df_future aos empreendimentos mais próximos em dados_empreend.

    Args:
        dados_empreend (pd.DataFrame): DataFrame com dados dos empreendimentos (lat, lon, CodGeracao, etc.).
        df_future (pd.DataFrame): DataFrame com previsões futuras de irradiância por LAT/LON/DATE.

    Returns:
        pd.DataFrame: DataFrame contendo os dados dos empreendimentos com irradiância prevista e distância.
    """
    print("Associando previsões de irradiância às usinas...")

    if 'data_prevista' not in df_future.columns:
        df_future['data_prevista'] = pd.to_datetime(df_future[['YEAR', 'MONTH', 'DAY']])

    # Cria o KDTree com as coordenadas do df_future para busca rápida
    # Use as coordenadas exatas presentes no df_future para a árvore
    tree_coords = df_future[['LAT', 'LON']].drop_duplicates().values
    tree = cKDTree(tree_coords)

    resultados = []
    
    # Mapear coordenadas originais de df_future para os índices da árvore
    # Isso permite que, após encontrar o índice, possamos recuperar a LAT/LON exata da grade
    df_future_unique_coords = df_future[['LAT', 'LON']].drop_duplicates().reset_index(drop=True)

    for i, row_empreend in dados_empreend.iterrows():
        lat_empreend, lon_empreend = row_empreend["lat"], row_empreend["lon"]

        # Encontrar o ponto mais próximo na árvore
        distancia, idx_ponto_na_arvore = tree.query([lat_empreend, lon_empreend])

        # Recuperar a LAT/LON da grade de previsão mais próxima
        closest_lat = df_future_unique_coords.iloc[idx_ponto_na_arvore]["LAT"]
        closest_lon = df_future_unique_coords.iloc[idx_ponto_na_arvore]["LON"]

        # Filtra todas as previsões para esse ponto exato na grade
        previsoes_local = df_future[
            (df_future["LAT"] == closest_lat) &
            (df_future["LON"] == closest_lon)
        ].copy()

        # Para cada data prevista daquele ponto, cria uma linha no resultado
        for _, linha_previsao in previsoes_local.iterrows():
            resultado = {
                'CodGeracao': row_empreend['CodGeracao'],
                'DataConexao': row_empreend['DataConexao'],
                'PotInstalada': row_empreend['PotInstalada'],
                'Municipio': row_empreend['Municipio'],
                'lat_empreendimento': lat_empreend, # Latitude original do empreendimento
                'lon_empreendimento': lon_empreend, # Longitude original do empreendimento
                'lat_previsao': closest_lat,      # Latitude do ponto de previsão usado
                'lon_previsao': closest_lon,      # Longitude do ponto de previsão usado
                'IRRADIANCE_FORECAST': linha_previsao['IRRADIANCE_FORECAST'],
                'DATA_PREVISTA': linha_previsao['data_prevista'],
                'DISTANCIA_KM': distancia * 111.32, # Aproximação para distância em km (1 grau ~ 111.32 km)
                'IRRADIANCE_YEAR': linha_previsao['YEAR'],
                'IRRADIANCE_MONTH': linha_previsao['MONTH'],
                'IRRADIANCE_DAY': linha_previsao['DAY']
            }
            resultados.append(resultado)

    return pd.DataFrame(resultados)


def calculate_generation(empreend_data_path, future_predictions_path, output_generation_path):
    """
    Calcula a geração de energia para as usinas com base nas previsões de irradiância.

    Args:
        empreend_data_path (str): Caminho para o CSV com dados dos empreendimentos (e.g., 'data/raw_data/dados_aneeltec_subsul.csv').
        future_predictions_path (str): Caminho para o CSV com as previsões futuras (e.g., 'data/processed_data/df_future_predictions.csv').
        output_generation_path (str): Caminho para salvar o DataFrame com os cálculos de geração.
    """
    print("Iniciando cálculos de geração de energia...")

    # Carregar dados dos empreendimentos
    dados_empreend = pd.read_csv(empreend_data_path, encoding="ISO-8859-1", sep=",")
    dados_empreend['PotInstalada'] = dados_empreend['PotInstalada'].str.replace(',', '.').astype(float)
    dados_empreend['lat'] = dados_empreend['lat'].astype(float)
    dados_empreend['lon'] = dados_empreend['lon'].astype(float)

    # Carregar previsões futuras de irradiância
    df_future = pd.read_csv(future_predictions_path)
    df_future['data_prevista'] = pd.to_datetime(df_future[['YEAR', 'MONTH', 'DAY']])

    # Associar previsões de irradiância às usinas
    df_final = associate_irradiance_to_plants(dados_empreend, df_future)

    # Converter DataConexao para datetime
    df_final['DataConexao'] = pd.to_datetime(df_final['DataConexao'], dayfirst=True, errors='coerce')
    df_final['DataConexao'] = df_final['DataConexao'].fillna(pd.to_datetime('2020-01-01')) # Preenche NaNs com uma data padrão

    # Definir data de referência para cálculo do tempo de operação (ex: último dia da previsão)
    # ou uma data fixa como 30 de junho de 2025 como no script original
    data_referencia_calculo = pd.to_datetime("2025-06-30") 

    df_final['TempoOperacao'] = (data_referencia_calculo - df_final['DataConexao']).dt.days / 365.25 # Considera ano bissexto
    df_final['TempoOperacao'] = df_final['TempoOperacao'].apply(lambda x: max(0, x)) # Garante que não há tempo de operação negativo

    # Parâmetros de cálculo de geração
    taxa_degradacao = 0.005 # 0.5% ao ano
    eficiencia_conversao = 0.25 # Exemplo: 25% de eficiência de conversão (irradiância para potência)

    df_final['FatorDegradacao'] = (1 - taxa_degradacao) ** df_final['TempoOperacao']

    # Cálculo da geração: Irradiância (W/m²) * Potência Instalada (kW) * FatorDegradacao * Eficiência
    # Ajuste das unidades: A irradiância é em W/m², PotInstalada em kW.
    # Para obter Geração em kWh/dia por kW instalado, precisamos considerar a base da irradiância.
    # Se IRRADIANCE_FORECAST for a média diária em W/m², multiplicar por 24h e dividir por 1000 para Wh -> kWh
    # Assumindo que IRRADIANCE_FORECAST já é a média diária em W/m² (ou uma medida representativa por dia)
    # E que o modelo prevê a irradiância para o dia inteiro, em W/m² (fluxo de energia)
    # Geração horária (Wh) = Irradiância (W/m²) * Área_painel (m²) * Eficiência
    # PotInstalada é em kW, logo já considera a área e a eficiência básica do painel
    # A fórmula no seu script: 'geracao' = 'IRRADIANCE_FORECAST' * 'PotInstalada' * 'FatorDegradacao'*eficiencia
    # Se IRRADIANCE_FORECAST for em W/m^2 (fluxo de potência), e PotInstalada em kW (potência máxima do sistema).
    # Multiplicar por 1 (hora) para converter de W/m² para Wh/m² (para um valor horário).
    # PotInstalada (kW) já incorpora a área e a eficiência do painel em sua capacidade nominal.
    # Se a previsão é diária em W/m², precisamos multiplicar por horas de sol efetivas ou converter W/m² diário em kWh/m² diário.
    # O cenário mais comum é que IRRADIANCE_FORECAST em W/m² seja um valor médio ou pico, e que a PotInstalada seja a capacidade nominal.
    # Para obter kWh gerados por dia, precisaríamos de Irradiância diária média (kWh/m²/dia) * Potência Instalada (kWp) / (Irradiância_STC * Fator_perdas).
    # A fórmula atual parece uma simplificação. Mantendo a lógica original do usuário, que sugere uma proporcionalidade direta.
    # Se IRRADIANCE_FORECAST é W/m² e PotInstalada é kW: 
    # Geração [kWh] = Irradiância [W/m²] * PotInstalada [kW] * FatorDegradacao * Eficiencia [adimensional] * (tempo_em_horas / 1000)
    # Se queremos em kWh/dia, e o modelo de previsão retorna um valor médio W/m2 para o dia, então:
    # Geracao (kWh/dia) = Irradiância_media_dia (W/m2) * Pot_Instalada (kW) * FatorDegradacao * Eficiencia (adimensional) * (24 horas / 1000 W/kW)
    # A sua fórmula original 'geracao' = 'IRRADIANCE_FORECAST' * 'PotInstalada' * 'FatorDegradacao'*eficiencia
    # Implica que IRRADIANCE_FORECAST seja um fator que multiplica a potência instalada, ou que as unidades já se alinham.
    # Para ser mais preciso e dado que a Irradiância é geralmente em W/m²:
    # Considerando PotInstalada em kWp (quilowatt-pico), Irradiância em W/m2 (potência de pico ou média horária).
    # Se a "IRRADIANCE_FORECAST" for a irradiância média para aquele dia em W/m², a geração diária em kWh seria:
    # Geração_Diária (kWh/dia) = (IRRADIANCE_FORECAST_W/m2 / 1000) * PotInstalada_kW * Fator_Degradacao * (Horas de Pico Equivalente ou fator de capacidade).
    # O "eficiencia" na sua fórmula pode ser um fator de capacidade ou perdas.
    # Vou manter a fórmula original que você forneceu, assumindo que as unidades se encaixam na sua intenção.
    # Se IRRADIANCE_FORECAST é a irradiância média *diária* em W/m², e queremos kWh/dia, precisamos converter para kWh.
    # 1 W/m² por um dia = 24 Wh/m² = 0.024 kWh/m².
    # Se PotInstalada já é em kW, podemos simplificar:
    # Geração (kWh/dia) = PotInstalada (kW) * (IRRADIANCE_FORECAST / 1000) * FatorDegradacao * Eficiencia
    # Isso faria mais sentido, mas seguirei sua fórmula literal:
    df_final['geracao'] = df_final['IRRADIANCE_FORECAST'] * df_final['PotInstalada'] * df_final['FatorDegradacao'] * eficiencia_conversao

    df_final['geracao'] = df_final['geracao'].apply(lambda x: max(0, x)) # Garante que a geração não é negativa

    df_final.to_csv(output_generation_path, index=False)
    print(f"Cálculos de geração concluídos e salvos em: {output_generation_path}")

if __name__ == "__main__":
    # Defina o caminho para os arquivos de entrada e saída
    EMPREEND_DATA_FILE = "data/raw_data/dados_aneeltec_subsul.csv"
    FUTURE_PREDICTIONS_FILE = "data/processed_data/df_future_predictions.csv"
    OUTPUT_GENERATION_FILE = "data/processed_data/df_final_generation.csv"

    # Certifique-se de que as pastas de dados existam
    os.makedirs(os.path.dirname(OUTPUT_GENERATION_FILE), exist_ok=True)

    calculate_generation(EMPREEND_DATA_FILE, FUTURE_PREDICTIONS_FILE, OUTPUT_GENERATION_FILE)