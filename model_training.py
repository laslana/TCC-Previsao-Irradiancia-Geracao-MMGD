# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, make_scorer
from xgboost import XGBRegressor, DMatrix, train as xgb_train
from scipy.stats import uniform, randint
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error (MAPE), tratando divisões por zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evitar divisão por zero ou valores muito pequenos que podem inflar o MAPE
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100 # Adiciona um pequeno epsilon

def train_irradiance_model(df_nasa_grouped_path, output_model_path, output_predictions_path,
                           start_date_forecast="2025-06-15", end_date_forecast="2025-06-21"):
    """
    Carrega dados processados, treina e otimiza um modelo XGBoost, e gera previsões futuras.

    Args:
        df_nasa_grouped_path (str): Caminho para o DataFrame agrupado (e.g., 'data/processed_data/df_nasa_grouped.csv').
        output_model_path (str): Caminho para salvar o modelo treinado (e.g., 'models/xgb_irradiance_model.json').
        output_predictions_path (str): Caminho para salvar as previsões futuras (e.g., 'data/processed_data/df_future_predictions.csv').
        start_date_forecast (str): Data de início para as previsões futuras (formato 'YYYY-MM-DD').
        end_date_forecast (str): Data de fim para as previsões futuras (formato 'YYYY-MM-DD').
    """
    print("Iniciando treinamento e otimização do modelo de irradiância...")
    df_nasa_grouped = pd.read_csv(df_nasa_grouped_path)
    df_nasa_grouped['DATE'] = pd.to_datetime(df_nasa_grouped['DATE'])

    # Features para o modelo
    X = df_nasa_grouped[["YEAR", "MONTH", "DAY", "LAT", "LON", "PRECTOTCORR", "T2M", "DAY_OF_YEAR", "LAT_LON"]]
    y = df_nasa_grouped["ALLSKY_SFC_SW_DWN"]

    # Divisão treino/validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo base
    xgb_base = XGBRegressor(objective='reg:squarederror', verbosity=0, random_state=42)

    # Espaço de hiperparâmetros
    param_dist = {
        'n_estimators': randint(300, 1000),
        'learning_rate': uniform(0.005, 0.3),
        'max_depth': randint(4, 20),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 10),
        'min_child_weight': randint(1, 10),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 4)
    }

    # RandomizedSearchCV
    print("Realizando busca de hiperparâmetros com RandomizedSearchCV (isso pode demorar)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=50,
        cv=3,
        verbose=0, # Reduz a verbosidade para não poluir o console
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"Melhores parâmetros encontrados: {best_params}")

    # Treinamento final com early stopping (usando DMatrix para otimização XGBoost)
    xgb_final = XGBRegressor(**best_params, objective='reg:squarederror', verbosity=0, random_state=42)
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)

    print("Treinando o modelo final com early stopping...")
    final_model = xgb_train(
        params=xgb_final.get_xgb_params(),
        dtrain=dtrain,
        num_boost_round=best_params['n_estimators'] + 100, # Um pouco mais que o n_estimators otimizado
        evals=[(dval, "validation")],
        early_stopping_rounds=50, # Aumentado para 50 para mais robustez
        verbose_eval=False
    )

    # Avaliação do modelo no conjunto de validação
    y_pred_val = final_model.predict(dval)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)

    print("\nMétricas de desempenho no conjunto de validação:")
    print(f"MAE: {mae_val:.4f}")
    print(f"R²: {r2_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"MAPE: {mape_val:.2f}%")

    # Geração de previsões futuras
    print("\nGerando previsões futuras de irradiância...")
    date_range = pd.date_range(start=start_date_forecast, end=end_date_forecast)
    df_base_coords = df_nasa_grouped[["LAT", "LON"]].drop_duplicates().copy()

    df_future = pd.DataFrame()
    for single_date in date_range:
        df_temp = df_base_coords.copy()
        df_temp["YEAR"] = single_date.year
        df_temp["MONTH"] = single_date.month
        df_temp["DAY"] = single_date.day
        df_temp["DATE"] = single_date # Adiciona a coluna DATE para DAY_OF_YEAR e LAT_LON
        df_future = pd.concat([df_future, df_temp], ignore_index=True)
    
    # Recalcula features de engenharia para df_future
    df_future["DAY_OF_YEAR"] = df_future["DATE"].dt.dayofyear
    df_future["LAT_LON"] = df_future["LAT"] * df_future["LON"]

    # Calcular médias históricas para PRECTOTCORR e T2M para simular dados futuros
    # Isso é uma simplificação. Em um cenário real, você precisaria de previsões meteorológicas futuras.
    media_historica = df_nasa_grouped.groupby(["LAT", "LON", "MONTH", "DAY"])[["PRECTOTCORR", "T2M"]].mean().reset_index()
    df_future = pd.merge(df_future, media_historica, on=["LAT", "LON", "MONTH", "DAY"], how="left")
    
    # Para garantir que não há NaNs nas features após o merge (pode ocorrer se a combinação LAT/LON/MONTH/DAY não existir nos dados históricos)
    df_future.fillna(df_nasa_grouped[["PRECTOTCORR", "T2M"]].mean(), inplace=True)

    X_future = df_future[["YEAR", "MONTH", "DAY", "LAT", "LON", "PRECTOTCORR", "T2M", "DAY_OF_YEAR", "LAT_LON"]]
    
    # Realizar previsões com o modelo otimizado
    # Assegura que X_future tenha as mesmas colunas que X_train e na mesma ordem
    X_future = X_future[X_train.columns] # Reordena colunas de X_future

    df_future["IRRADIANCE_FORECAST"] = final_model.predict(DMatrix(X_future))

    # Salvar previsões futuras
    df_future.to_csv(output_predictions_path, index=False)
    print(f"Previsões futuras de irradiância salvas em: {output_predictions_path}")
    print("Treinamento e previsão concluídos com sucesso!")

if __name__ == "__main__":
    # Defina o caminho para o DataFrame processado de entrada e os arquivos de saída
    PROCESSED_DATA_FILE = "data/processed_data/df_nasa_grouped.csv"
    OUTPUT_MODEL_FILE = "models/xgb_irradiance_model.json" # Salvando como JSON para compatibilidade
    FUTURE_PREDICTIONS_FILE = "data/processed_data/df_future_predictions.csv"
    
    # Certifique-se de que a pasta de modelos exista
    os.makedirs(os.path.dirname(OUTPUT_MODEL_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(FUTURE_PREDICTIONS_FILE), exist_ok=True)

    # Você pode definir as datas de previsão aqui
    START_DATE = "2025-07-01" # Exemplo: início de julho de 2025
    END_DATE = "2025-07-31"   # Exemplo: fim de julho de 2025

    train_irradiance_model(PROCESSED_DATA_FILE, OUTPUT_MODEL_FILE, FUTURE_PREDICTIONS_FILE,
                           start_date_forecast=START_DATE, end_date_forecast=END_DATE)
    
    # Para salvar o modelo separadamente, caso o usuário queira recarregar
    # model_otimizado.save_model(OUTPUT_MODEL_FILE)