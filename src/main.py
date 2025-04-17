from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.src.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from CONSTANTS import *
from TEXTS import *
import os
import pandas as pd
import joblib
import json
import keras

class ProgressBarCallback(Callback):
    def __init__(self, epochs, window):
        super().__init__()
        self.progress_bar = window.progress_bar
        self.label_var = window.label_var
        self.stage_var = window.stage_var
        self.epochs = epochs
        self.window = window

        self.restart(FIRST_TRAINING_STAGE)

    def restart(self, stage_text):
        self.progress_bar['value'] = 0
        self.stage_var.set(stage_text)

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs * 100
        self.progress_bar['value'] = progress
        self.label_var.set(TRAINING_PROGRESS_LABEL_TEMPLATE.format(epoch+1, self.epochs))
        self.window.update_idletasks()

def get_models() -> dict[str:dict]:

    '''
    Возвращает модели по файлу описания моделей
            Возвращаемое значение:
                    models (dict[str:dict]): словарь моделей, ключ - ID, значение - словарь со свойствам
    '''

    models = {
        STANDARD_MODEL_ID: {"name":STANDARD_MODEL_NAME, "var_name":None},
    }

    descriptions = {}

    try:

        with open(MODELS_DESCRIPTION_FILENAME) as f:
            descriptions = json.load(f)

        for model_id, description in descriptions.items():
            description["var_name"] = None
            models[model_id] = description

    except FileNotFoundError:
        # Ничего не делаем, если файла нет
        pass

    return models

def get_model(id: str) -> tuple[keras.src.models.model, MinMaxScaler, MinMaxScaler]:

    '''
    Возвращает кортеж: (модель, Scaler для X, Scaler для Y)
            Параметры:
                    id (str): id модели
            Возвращаемое значение:
                    model(tuple[keras.src.models.model, MinMaxScaler, MinMaxScaler]): необходимые модели
    '''

    model = None
    scaler_X = None
    scaler_Y = None

    model_path = id + ".keras"
    scaler_X_path = "scaler_X" + id + ".keras"
    scaler_Y_path = "scaler_Y" + id + ".keras"

    for model_file in os.scandir(MODELS_DIRECTORY_PATH):  
        
        if model_file.is_file():
            
            if model_file.name == model_path:
                model = load_model(MODELS_DIRECTORY_PATH + "\\" + model_path)
            if model_file.name == scaler_X_path:
                scaler_X = joblib.load(MODELS_DIRECTORY_PATH + "\\" + scaler_X_path)
            if model_file.name == scaler_Y_path:
                scaler_Y = joblib.load(MODELS_DIRECTORY_PATH + "\\" + scaler_Y_path)

            if model and scaler_X and scaler_Y:
                break
            
    return model, scaler_X, scaler_Y

def get_standard_model() -> keras.src.models.model:

    '''
    Возвращает стандартную заготовленную модель
            Возвращаемое значение:
                    model(keras.src.models.model): модель
    '''

    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

def get_sсaled_data(years:list[int], months_sin:list[int], months_cos:list[int], temps:list[int], scaler_X:MinMaxScaler, scaler_y:MinMaxScaler) -> tuple[np.array, np.array]:
  
    '''
    Возвращает нормализованные данные
            Параметры:
                    years(list[int]): года
                    months(list[int]): месяца
                    temps(list[int]): temps[i] = средняя температура years[i], month[i]
                    scaler_X(MinMaxScaler): модель для нормализации годов и месяцев
                    scaler_Y(MinMaxScaler): модель для нормализации температур
            Возвращаемое значение:
                    scaled_data(tuple[np.array, np.array]): нормализованные данные
    '''

    y = temps

    X_years = np.array(years).reshape(len(years), 1)
    X_years = scaler_X.fit_transform(X_years)
    X_scaled = np.column_stack([X_years.flatten(), months_sin, months_cos])
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled

def train_model(model_id:str, train_data_filename:str, epochs:int, window):

    '''
    Тренирует модель
            Параметры:
                    model_id(str): id модели
                    train_data_filename(str): имя файла, на котором происходит обучение
                    epochs(int): количество эпох
            Возвращаемое значение:
                    train_data(tuple[dict, keras.src.models.model, dict, dict]): передает history обучения, обученную модель, результаты оценки (mae, mse), модели нормализации
    '''

    find_model = False
    model = None

    model_filename = model_id + ".keras"

    for model_file in os.scandir(MODELS_DIRECTORY_PATH):  
        if model_file.is_file() and model_file.name == model_filename:
            model = load_model(MODELS_DIRECTORY_PATH + "\\" + model_filename)
            find_model = True

    if not find_model:
        # Если не нашли стандартную модель в файлах, значит нужно ее создать
        if model_id == STANDARD_MODEL_ID:
            model = get_standard_model()
        # Если не нашли нестандартную модель в файлах, значит ее кто-то удалил оттуда
        else:
            raise FileNotFoundError("Не найден файл модели")

    historic_data = pd.read_csv(os.path.abspath(train_data_filename), encoding='utf-8', sep=";", index_col=False, header=0)
    if "timestamp" not in historic_data.columns:
        historic_data = historic_data.rename(columns={'Местное время в Москве (ВДНХ)':'timestamp'})

    historic_data = historic_data[["timestamp", "T"]]
    historic_data['timestamp'] = pd.to_datetime(historic_data['timestamp'], dayfirst=True)
    historic_data['year'] = historic_data.timestamp.dt.year
    historic_data['month'] = historic_data.timestamp.dt.month

    aggregated_data = historic_data.groupby(['year', 'month'])['T'].agg(['mean']).reset_index()
    aggregated_data['month_sin'] = np.sin(np.pi * (aggregated_data['month']-1) / 11)
    aggregated_data['month_cos'] = np.cos(np.pi * (aggregated_data['month']-1) / 11)

    years = aggregated_data["year"]

    temperatures = np.array(aggregated_data["mean"])

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled, y_scaled = get_sсaled_data(years, aggregated_data['month_sin'], aggregated_data['month_cos'], temperatures, scaler_X, scaler_y)

    # Оцениваем точность модели
    evaluate_test = {"mae":0, "mse":0}
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
    progress_callback = ProgressBarCallback(epochs, window)
    model.fit(X_train, y_train, epochs=epochs, callbacks=[progress_callback])
    progress_callback.restart(SECOND_TRAINING_STAGE)

    predictions = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(predictions)
    y_true = scaler_y.inverse_transform(y_test)
    evaluate_test["mae"] = mean_absolute_error(y_true, y_pred)
    evaluate_test["mse"] = mean_squared_error(y_true, y_pred)

    progress_callback.restart(THIRD_TRAINING_STAGE)

    # Обучение модели
    history = model.fit(X_test, y_test, epochs=epochs, verbose=1, callbacks=[progress_callback])
    

    return history, model, evaluate_test, {"scaler_X": scaler_X, "scaler_Y": scaler_y}