import customtkinter as ctk
from PIL import Image
import os
from keras.models import load_model
from CONSTANTS import *
from TEXTS import *
from main import get_models, train_model, get_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import keras
import joblib
import numpy as np
import uuid
import json
import types
from CTkMessagebox import CTkMessagebox
from tkinter import ttk

class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_WINDOW_TITLE)
        self.geometry(f"{800}x{400}")
        self.minsize(700, 300)
        self.maxsize(850, 600)

        logo_image = ctk.CTkImage(dark_image=Image.open(LOGO_IMAGE_PATH),
                                    size=(30, 30))

        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text=APP_TITLE, font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=1, padx=(1, 10), pady=(20, 10))

        self.logo = ctk.CTkLabel(self.sidebar_frame, text="", image=logo_image)
        self.logo.grid(row=0, column=0, padx=5, pady=(20, 10))

        self.tabview = ctk.CTkTabview(self, height=700)
        self.tabview.grid(row=0, column=1, padx=(2, 2), pady=(2, 0), sticky="nsew")
        self.tabview.add(MODELS_TAB_NAME)

        self.models = get_models()

        self.models_tab = self.tabview.tab(MODELS_TAB_NAME)
        self.models_elements = []

        self.draw_models_table()

    def draw_models_table(self):

        '''
        Заново перерисовывает таблицу моделей по self.models
        '''

        change_name_image = ctk.CTkImage(dark_image=Image.open(CHANGE_NAME_IMAGE_PATH))

        if len(self.models_elements) > 0:
            for element in self.models_elements:
                element.destroy()

        self.models_elements = []

        for row, (model_id, model) in enumerate(self.models.items()):
            
            column = 0

            entry_var = ctk.StringVar(name=model_id)
            entry = ctk.CTkEntry(self.models_tab, textvariable=entry_var, state="readonly")
            model["var_name"] = entry_var
            entry_var.set(model["name"])
            entry.grid(row=row, column=column, 
                    padx=10, pady=5)
            
            self.models_elements.append(entry)
            
            column += 1
            
            change_name_button = ctk.CTkButton(self.models_tab, width=20, text="", image=change_name_image, 
                                               command=self.get_command_change_name_button(model_id), 
                                               state=ctk.DISABLED if (model_id == STANDARD_MODEL_ID) else ctk.NORMAL)
            change_name_button.grid(row=row, column=column, padx=5)

            self.models_elements.append(change_name_button)

            column += 1

            train_model = ctk.CTkButton(self.models_tab, width=20, text=TRAIN_BUTTON_TEXT, 
                                        command=self.get_command_train_button(model_id),
                                        state=ctk.DISABLED if (model_id == STANDARD_MODEL_ID) else ctk.NORMAL)
            train_model.grid(row=row, column=column, padx=10)

            self.models_elements.append(train_model)

            column += 1

            predict_temps = ctk.CTkButton(self.models_tab, width=20, text=PREDICT_BUTTON_TEXT, command=self.get_command_predict_button(model_id))
            predict_temps.grid(row=row, column=column, padx=10)

            self.models_elements.append(predict_temps)

            column += 1

            delete_model = ctk.CTkButton(self.models_tab, width=20, text=DEELTE_MODEL_BUTTON_TEXT, 
                                        command=self.get_command_delete_button(model_id),
                                        state=ctk.DISABLED if (model_id == STANDARD_MODEL_ID) else ctk.NORMAL)
            delete_model.grid(row=row, column=column, padx=10)

            self.models_elements.append(delete_model)

        row += 1

        load_model_button = ctk.CTkButton(self.models_tab, text=LOAD_MODEL_BUTTON_TEXT, command=self.load_model)
        load_model_button.grid(row=row, column=0, columnspan=3, padx=10, pady=8, sticky="w")

        self.models_elements.append(load_model_button)

        return row

    def get_command_change_name_button(self, id:str) -> types.FunctionType:

        '''
        Возвращает функцию, которая открывает диалог изменения наименования модели
                Параметры:
                        id (str): id модели, которой нужно будет изменять наименование при нажатии на кнопку
                Возвращаемое значение:
                        push_change_name (types.FunctionType): функция
        '''

        def push_change_name():
            dialog = ctk.CTkInputDialog(text=CHANGE_NAME_WINDOW_TEXT, title=CHANGE_NAME_WINDOW_TITLE)
            new_name = dialog.get_input()
            if new_name != None:
                self.models[id]["name"] = new_name
                self.models[id]["var_name"].set(new_name)
        
        return push_change_name
    
    def get_command_save_button(self, model:keras.src.models.model, window) -> types.FunctionType:

        '''
        Возвращает функцию, которая открывает диалог сохранения модели
                Параметры:
                        model (keras.src.models.model): модель, которую нужно будет сохранить по нажатию на кнопку
                Возвращаемое значение:
                        push_save_model (function): функция
        '''

        def push_save_model():
            filetypes = (("Файл модели (*.keras)", "*.keras"),
                        ("Любой", "*"))
            window.attributes('-topmost', False)
            new_filename = ctk.filedialog.asksaveasfilename(filetypes=filetypes, title=SAVE_MODEL_WINDOW_TITLE)
            window.attributes('-topmost', True)
            if not new_filename:
                return
            if not str.endswith(new_filename, ".keras"):
                new_filename += ".keras"
            model.save(new_filename)
        
        return push_save_model
    
    def get_command_train_button(self, id:str) -> types.FunctionType:

        '''
        Возвращает функцию, которая открывает диалог обучения модели
                Параметры:
                        id (str): id модели, которой нужно будет обучать при нажатии на кнопку
                Возвращаемое значение:
                        push_train_model (types.FunctionType): функция
        '''

        def push_train_model():
            
            ask_about_training(self, id)

        return push_train_model
    
    def get_command_predict_button(self, id:str)  -> types.FunctionType:

        '''
        Возвращает функцию, которая открывает окно предсказания модели
                Параметры:
                        id (str): id модели, которая будет предсказывать результат при нажатии на кнопку
                Возвращаемое значение:
                        push_predict (types.FunctionType): функция
        '''

        def push_predict():

            def ask_about_year():

                '''
                Задает вопрос: будет ли для обучения использовать свой файл или стандартный
                        Параметры:
                                window (ctk.CTk): главное окно
                                id (str): id модели, которая будет удалена при нажатии на кнопку
                '''

                def after_answer():
                    
                    year = v.get()
                    if not (year.isdigit()) or int(year) < 1 or int(year) > 3000:
                        CTkMessagebox(message=INCORRECT_YEAR_ERROR_TEXT)
                        return
                    else:
                        year_window.destroy()
                        year_window.update()
                        predict(year)

                def predict(year):

                    def save_predictions():

                        filetypes = (("Текстовый файл (*.txt)", "*.txt"),
                                ("Любой", "*"))

                        prediction_result_window.attributes('-topmost', False)
                        new_filename = ctk.filedialog.asksaveasfilename(filetypes=filetypes, title=SAVE_RESULT_WINDOW_TITLE)
                        prediction_result_window.attributes('-topmost', True)

                        if not new_filename:
                            return

                        if not str.endswith(new_filename, ".txt"):
                            new_filename += ".txt"

                        with open(new_filename,'w') as f:
                            f.write('Месяц;Температура\n')
                            for month in range(0, 12):
                                transfer = "" if month == 12 else "\n"
                                f.write(f'{MONTHS_NAME[month]};{predicted_temperature[month]:.3f}{transfer}')

                    model, scaler_X, scaler_Y = get_model(id)

                    if model is None:
                        CTkMessagebox(title="Info", message=STANDARD_MODEL_ISNT_READY)
                        return

                    test_years = np.array([[int(year)] for _ in range(0, 12)])
                    test_months = np.array([i for i in range(1, 13)])
                    months_sin = np.sin(np.pi * (test_months-1) / 11)
                    months_cos = np.cos(np.pi * (test_months-1) / 11)
                    
                    if scaler_X is None:
                        # Нет scaler значит, что на вход даются ненормализованные числа
                        test_input_scaled = test_years.flatten()
                    else:
                        test_input_scaled = scaler_X.transform(test_years).flatten()

                    predicted_temperature_scaled = model.predict(np.column_stack([test_input_scaled, months_sin, months_cos]))

                    if scaler_Y is None:
                        # Нет scaler значит, что на выход выдаются ненормализованные числа
                        predicted_temperature = predicted_temperature_scaled.flatten()
                    else:
                        predicted_temperature = scaler_Y.inverse_transform(predicted_temperature_scaled).flatten()

                    prediction_result_window = ctk.CTkToplevel(self)
                    prediction_result_window.title(PREDICT_WINDOW_TITLE_TEMPLATE.format(year))
                    for month in range(0, 12):
                        label = ctk.CTkLabel(prediction_result_window, width=10, text=f"{MONTHS_NAME[month]}:")
                        label.grid(row=month, column=0, padx=10, sticky="w")
                        label = ctk.CTkLabel(prediction_result_window, text=f"{predicted_temperature[month]:.3f}")
                        label.grid(row=month, column=1, padx=10)

                    y = predicted_temperature
                    x = MONTHS_NAME
                    fig = Figure(figsize=(13, 8), dpi=70)
                    ax = fig.add_subplot(111)
                    ax.bar(x, y)
                    canvas = FigureCanvasTkAgg(fig, master=prediction_result_window)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=0, rowspan=15, column=2, columnspan=6, padx=5, pady=5)

                    row = month + 1
                    save_result_button = ctk.CTkButton(prediction_result_window, text=SAVE_RESULT_BUTTON, command=save_predictions)
                    save_result_button.grid(row=row, column=0, columnspan=2, padx=10, pady=5)

                    row = row + 1
                    OK_button = ctk.CTkButton(prediction_result_window, text=OK_BUTTON_TEXT, command=prediction_result_window.destroy)
                    OK_button.grid(row=row, column=0, columnspan=2, padx=10, pady=5)

                    prediction_result_window.resizable(False, False)

                    prediction_result_window.attributes('-topmost', True)


                year_window = ctk.CTkToplevel(self)
                year_window.title(YEAR_PREDICT_WINDOW_TITLE)
                ctk.CTkLabel(year_window, text=YEAR_PREDICT_LABEL_TEXT).pack(ipadx=10, padx=10)

                v = ctk.StringVar()

                ctk.CTkEntry(year_window, textvariable=v).pack(ipadx=10, padx=10)

                ctk.CTkButton(year_window, text=OK_BUTTON_TEXT, command=after_answer).pack(pady=10)
                year_window.attributes('-topmost', True)

            ask_about_year()
            return

        return push_predict

    def get_command_delete_button(self, id:str) -> types.FunctionType:


        '''
        Возвращает функцию, которая удаляет модель при нажатии на кнопку
                Параметры:
                        id (str): id модели, которая будет удалена при нажатии на кнопку
                Возвращаемое значение:
                        push_delete (types.FunctionType): функция
        '''
        
        def push_delete():

            del self.models[id]
            model_path = id + ".keras"
            scaler_X_path = "scaler_X" + id + ".keras"
            scaler_Y_path = "scaler_Y" + id + ".keras"

            find_model = False
            find_scaler_X = False
            find_scaler_y = False

            for model_file in os.scandir(MODELS_DIRECTORY_PATH):  
                    
                if model_file.is_file():
                    
                    if model_file.name == model_path:
                        os.remove(MODELS_DIRECTORY_PATH + "\\" + model_path)
                        find_model = True
                    if model_file.name == scaler_X_path:
                        os.remove(MODELS_DIRECTORY_PATH + "\\" + scaler_X_path)
                        find_scaler_X = True
                    if model_file.name == scaler_Y_path:
                        os.remove(MODELS_DIRECTORY_PATH + "\\" + scaler_Y_path)
                        find_scaler_y = True

                    if find_model and find_scaler_X and find_scaler_y:
                        break
            
            self.draw_models_table()
            save_models_description(self.models)

        return push_delete

    def load_model(self):

        '''
        Загружает модель, которую укажет пользователь
        '''

        filetypes = (("Файл модели (*.keras)", "*.keras"),
                    ("Любой", "*"))
        filename = ctk.filedialog.askopenfilename(filetypes=filetypes, title=LOAD_MODEL_WINDOW_TITLE)
        if not filename:
            return
        model = load_model(filename)
        new_id = str(uuid.uuid4())
        model.save(MODELS_DIRECTORY_PATH + "\\" + new_id + ".keras")

        dialog = ctk.CTkInputDialog(text=CHANGE_NAME_WINDOW_TEXT, title=CHANGE_NAME_WINDOW_TITLE)
        new_name = dialog.get_input()
        if new_name == None:
            new_name = NO_NAME_MODEL_NAME

        new_model = {}
        new_model["name"] = new_name
        self.models[new_id] = new_model

        # [?] Думаю не стоит перерисовать таблицу полностью при добавлении новой модели
        self.draw_models_table()
        save_models_description(self.models)

def ask_about_training(window:ctk.CTk, id:str):

    '''
    Задает вопрос: будет ли для обучения использовать свой файл или стандартный
            Параметры:
                    window (ctk.CTk): главное окно
                    id (str): id модели, которая будет удалена при нажатии на кнопку
    '''

    def after_answer():

        epochs_text = epoch_var.get()

        if epochs_text.isdigit() and int(epochs_text) > 0 and int(epochs_text) < 1000:
            epochs = int(epochs_text)
        else:
            CTkMessagebox(message=INCORRECT_EPOCHS_ERROR_TEXT)
            return
        
        root.destroy()
        root.update()
        mode = v.get()
        if mode == FILE_TRAIN_MODE:
            filetypes = (("Текстовый файл (*.txt, *.csv)", "*.txt *.csv"),
                        ("Любой", "*"))
            filename = ctk.filedialog.askopenfilename(filetypes=filetypes, title=TRAIN_FILE_WINDOW_TITLE)

        elif mode == STANDARD_TRAIN_MODE:
            filename = STANDARD_TRAIN_DATA_PATH

        if filename:
            show_train_window(id, filename, window, epochs)

    root = ctk.CTkToplevel(window)

    root.title(TRAIN_MODE_WINDOW_TITLE)
    ctk.CTkLabel(root, text=TRAIN_MODE_QUESTION).pack(ipadx=10, padx=10, pady=5)

    v = ctk.IntVar()
    epoch_var = ctk.StringVar(value=DEFAULT_EPOCHS)

    train_options = [
        {"text": STANDARD_TRAIN_MODE_TITLE, "value": STANDARD_TRAIN_MODE},
        {"text": FILE_TRAIN_MODE_TITLE, "value": FILE_TRAIN_MODE}
    ]

    ctk.CTkLabel(root, text=TRAIN_WINDOW_EPOCHS_SUBTITLE).pack(anchor="w", ipadx=10, padx=10)
    ctk.CTkEntry(root, textvariable=epoch_var).pack(anchor="w", ipadx=10, padx=10, pady=5)

    for option in train_options:
        ctk.CTkRadioButton(root, text=option["text"], variable=v, value=option["value"]).pack(anchor="w", padx=7, pady=3)

    ctk.CTkButton(root, text=OK_BUTTON_TEXT, command=after_answer).pack(pady=10)
    root.attributes('-topmost', True)

def show_train_window(id:str, filename:str, window:ctk.CTk, epochs:int):

    '''
    Открывает окно обучения модели
            Параметры:
                    id (str): id модели, которая будет удалена при нажатии на кнопку
                    filename(str): имя файла для обучения
                    window (ctk.CTk): главное окно  
                    epochs (int): количество эпох                  
    '''

    def begin_train():

        history, model, evaluate_res, scalers = train_model(id, filename, epochs, root)
        model_path = MODELS_DIRECTORY_PATH + "\\" + id + ".keras"
        model.save(model_path)
        joblib.dump(scalers["scaler_X"], MODELS_DIRECTORY_PATH + "\\" + "scaler_X" + id + ".keras") 
        joblib.dump(scalers["scaler_Y"], MODELS_DIRECTORY_PATH + "\\" + "scaler_Y" + id + ".keras")

        y = history.history["loss"]
        x = [i for i in range(1, epochs+1)]
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        ctk.CTkLabel(root, text=LOSS_CHART_SUBTITLE).pack(ipadx=10, padx=10)
        ctk.CTkLabel(root, text=EVALUATE_RESULT_TEXT.format(evaluate_res["mae"], evaluate_res["mse"])).pack(ipadx=10, padx=10)
        ctk.CTkButton(root, text=OK_BUTTON_TEXT, command=root.destroy).pack(pady=5, side="left", padx=5)
        ctk.CTkButton(root, text=SAVE_BUTTON_TEXT, command=window.get_command_save_button(model, root)).pack(pady=5, side="left", padx=5)
        label.configure(text=TRAIN_DONE_LABEL_TEXT)
        root.progress_bar.pack_forget()
        root.progress_label.pack_forget()
        root.stage_label.pack_forget()
        root.geometry(f"{430}x{500}")

    root = ctk.CTkToplevel(window)
    root.title(TRAIN_WINDOW_TITLE)
    label = ctk.CTkLabel(root, text=TRAIN_WAITING_LABEL_TEXT)
    label.pack(ipadx=10, padx=10, pady=10)
    root.progress_bar = ttk.Progressbar(master=root, orient="horizontal", length=440, value=0)
    root.progress_bar.pack(pady=10)
    root.label_var = ctk.StringVar()
    root.stage_var = ctk.StringVar()
    root.stage_label = ctk.CTkEntry(master=root, textvariable=root.stage_var, width=300, justify="center")
    root.stage_label.pack(pady=5)
    root.progress_label = ctk.CTkEntry(master=root, textvariable=root.label_var, width=300, justify="center")
    root.progress_label.pack(pady=5)

    root.attributes('-topmost', True)

    root.geometry(f"{430}x{250}")
    root.resizable(False, False)

    root.after(200, begin_train)

def save_models_description(models:dict[str:dict[str:str]]):

    '''
    Сохраняет описания моделей в файл
            Параметры:
                    models (dict[str:dict[str:str]]): App.models                
    '''

    descriptions = {}
    for model_id, description in models.items():
        if model_id != STANDARD_MODEL_ID:
            dump_description = {"name": description["name"]}
            descriptions[model_id] = dump_description
    
    with open(MODELS_DESCRIPTION_FILENAME, 'w') as f:
        json.dump(descriptions, f)

def restore_integrity():

    '''
    Восстанавливает необходимые файлы, если они были удалены              
    '''

    if not os.path.exists(MODELS_DIRECTORY_PATH):
        os.makedirs(MODELS_DIRECTORY_PATH)

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")
    restore_integrity()
    app = App()
    app.mainloop()