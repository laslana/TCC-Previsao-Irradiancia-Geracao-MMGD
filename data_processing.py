# src/data_processing.py

import pandas as pd
import numpy as np
import glob
from sklearn.cluster import KMeans

def carregar_csvs_nasa(lista_arquivos):
    """
    Carrega arquivos CSV do NASA POWER, ignorando cabeçalhos até a linha de dados.
    """
    dataframes = []
    for arquivo in lista_arquivos:
        try:
            with open(arquivo, 'r') as f:
                linhas = f.readlines()
            skip = 0
            for i, linha in enumerate(linhas):
                if linha.startswith("LON") or linha.startswith("YEAR") or "LON" in linha.upper():
                    skip = i
                    break
            df = pd.read_csv(arquivo, skiprows=skip)
            dataframes.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()

def process_nasa_data(raw_data_path, output_path):
    """
    Carrega, corrige e agrupa dados de irradiância, temperatura e precipitação da NASA POWER.

    Args:
        raw_data_path (str): Caminho base para os dados brutos (e.g., 'data/raw_data/').
        output_path (str): Caminho para salvar o DataFrame processado (e.g., 'data/processed_data/df_nasa_grouped.csv').
    """
    print("Iniciando processamento de dados NASA POWER...")

    # Carregar dados de irradiância
    # Assumindo que 'irrad sub sul/' contém os CSVs de irradiância
    arquivos_irradiancia = glob.glob(f"{raw_data_path}irrad_sub_sul/*.csv")
    df_irradiancia_raw = carregar_csvs_nasa(arquivos_irradiancia)

    # Correção de valores -999.0 usando KMeans para irradiância
    df_irradiancia = df_irradiancia_raw.copy()
    if -999.0 in df_irradiancia["ALLSKY_SFC_SW_DWN"].unique():
        print("Corrigindo valores -999.0 na irradiância com KMeans...")
        df_validos = df_irradiancia[df_irradiancia["ALLSKY_SFC_SW_DWN"] != -999.0].copy()
        
        if not df_validos.empty:
            # Determine num_clusters com base em valores únicos de LAT/LON nos dados válidos
            unique_coords = df_validos[["LAT", "LON"]].drop_duplicates()
            num_clusters = min(len(unique_coords), 42) # Limite máximo de 42 se houver mais

            if num_clusters > 0:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                df_validos["CLUSTER"] = kmeans.fit_predict(df_validos[["LAT", "LON"]])
                media_por_cluster = df_validos.groupby("CLUSTER")["ALLSKY_SFC_SW_DWN"].mean()
                
                # Prever clusters para todo o dataframe original
                df_irradiancia["CLUSTER"] = kmeans.predict(df_irradiancia[["LAT", "LON"]])
                df_irradiancia.loc[df_irradiancia["ALLSKY_SFC_SW_DWN"] == -999.0, "ALLSKY_SFC_SW_DWN"] = \
                    df_irradiancia["CLUSTER"].map(media_por_cluster)
                df_irradiancia.drop(columns=["CLUSTER"], inplace=True)
                print("Correção de irradiância concluída.")
            else:
                print("Não há dados válidos suficientes para aplicar KMeans para correção de irradiância.")
        else:
            print("Não há dados válidos para correção de irradiância. Ignorando correção.")
    else:
        print("Não há valores -999.0 na irradiância. Correção não necessária.")

    # Carregar dados de temperatura
    arquivos_temp = glob.glob(f"{raw_data_path}DADOS_TEMP_SUBSUL/**/*.csv", recursive=True)
    df_temp = carregar_csvs_nasa(arquivos_temp)

    # Carregar dados de precipitação
    arquivos_precip = glob.glob(f"{raw_data_path}DADOS_PREC_SUBSUL/**/*.csv", recursive=True)
    df_precip = carregar_csvs_nasa(arquivos_precip)

    # Juntar os DataFrames
    print("Mesclando DataFrames de temperatura, precipitação e irradiância...")
    df_merged = pd.merge(df_temp, df_precip, how='inner', on=['LAT', 'LON', 'YEAR', 'MO', 'DY'])
    df_final_raw = pd.merge(df_merged, df_irradiancia, how='inner', on=['LAT', 'LON', 'YEAR', 'MO', 'DY'])

    # Agrupar e renomear colunas
    print("Agrupando dados por dia e coordenadas...")
    df_nasa_grouped = df_final_raw.groupby(["LAT", "LON", "YEAR", "MO", "DY"]).agg({
        "ALLSKY_SFC_SW_DWN": "mean",
        "PRECTOTCORR": "mean",
        "T2M": "mean"
    }).reset_index()

    df_nasa_grouped.rename(columns={"MO": "MONTH", "DY": "DAY"}, inplace=True)
    df_nasa_grouped["DATE"] = pd.to_datetime(df_nasa_grouped[["YEAR", "MONTH", "DAY"]])
    
    # Adicionar outras features de engenharia
    df_nasa_grouped["DAY_OF_YEAR"] = df_nasa_grouped["DATE"].dt.dayofyear
    df_nasa_grouped["LAT_LON"] = df_nasa_grouped["LAT"] * df_nasa_grouped["LON"]

    # Salvar o DataFrame processado
    df_nasa_grouped.to_csv(output_path, index=False)
    print(f"Dados processados salvos em: {output_path}")
    print("Processamento de dados concluído com sucesso!")

if __name__ == "__main__":
    # Defina o caminho para a pasta de dados brutos e para o arquivo de saída
    RAW_DATA_DIR = "data/raw_data/"
    PROCESSED_DATA_FILE = "data/processed_data/df_nasa_grouped.csv"
    
    # Certifique-se de que as pastas de dados existam
    import os
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)

    process_nasa_data(RAW_DATA_DIR, PROCESSED_DATA_FILE)