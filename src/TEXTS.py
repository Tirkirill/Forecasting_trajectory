
# Главное окно
APP_WINDOW_TITLE = "Предсказание погоды"
APP_TITLE = "Прогнозатор"
STANDARD_MODEL_ISNT_READY = "Стандартная модель пока не обучена"

# Окно выбора режима обучения (свой файл или стандартный набор)
STANDARD_TRAIN_MODE_TITLE = "Использовать стандартный набор"
FILE_TRAIN_MODE_TITLE = "Использовать свой файл"
TRAIN_MODE_QUESTION = "Вы хотите использовать стандартный набор для обучения или использовать свой файл?\n" \
"Свой файл должен иметь 2 колонки: 'timestamp' (или 'Местное время в Москве (ВДНХ)') и 'T' (температура)"
TRAIN_MODE_WINDOW_TITLE = "Обучение модели"

# Окно выбора файла обучения
TRAIN_FILE_WINDOW_TITLE = "Выберите файл для обучения"

# Окно сохранения модели
SAVE_MODEL_WINDOW_TITLE = "Сохранения файла модели"

# Окно изменения имени модели
CHANGE_NAME_WINDOW_TITLE = "Новое имя"
CHANGE_NAME_WINDOW_TEXT = "Новое имя модели:"

# Окно обучения
TRAIN_WINDOW_TITLE = "Обучение модели"
TRAIN_WAITING_LABEL_TEXT = "Происходит обучение модели. Пожалуйста ожидайте"
TRAIN_DONE_LABEL_TEXT = "Спасибо за ожидание. Обучение завершено. \n" \
"Обученная модель сохранена во внутренних файлах программы.\n" \
"Вы можете дополнительно сохранить ее на жесткий диск"
LOSS_CHART_SUBTITLE = "График потерь"
EVALUATE_RESULT_TEXT = "Средняя абсолютная ошибка: {0}\nСредняя квадратичная ошибка: {1}"
TRAIN_WINDOW_EPOCHS_SUBTITLE = "Количество эпох:"
INCORRECT_EPOCHS_ERROR_TEXT = "Требуется ввести число эпох от 1 до 1000"
TRAINING_PROGRESS_LABEL_TEMPLATE = "Прошло {0} эпох из {1}"
FIRST_TRAINING_STAGE = "Обучаем модель на тренировочных данных"
SECOND_TRAINING_STAGE = "Оцениваем модель"
THIRD_TRAINING_STAGE = "Обучаем модель на валидационных данных"

# Окно ввода года предсказания
YEAR_PREDICT_WINDOW_TITLE = "Год"
YEAR_PREDICT_LABEL_TEXT = "Введите год, на который хотите получить прогноз"
INCORRECT_YEAR_ERROR_TEXT = "Требуется ввести год от 2025 до 2030"

# Окно предсказания
PREDICT_WINDOW_TITLE_TEMPLATE = "Погода на {0} год"
SAVE_RESULT_BUTTON = "Сохранить результат"
SAVE_RESULT_WINDOW_TITLE = "Результат"
MODEL_ISNT_READY_ERROR_TEXT = "Модель еще не обучена"
MODEL_ISNT_READY_ERROR_TITLE = "Модель не готова"

# Кнопка "OK"
OK_BUTTON_TEXT = "Готово"

# Кнопки для моделей
TRAIN_BUTTON_TEXT = "Обучить"
SAVE_BUTTON_TEXT = "Сохранить"
PREDICT_BUTTON_TEXT = "Получить результат"
LOAD_MODEL_BUTTON_TEXT = "Загрузить новую модель"
LOAD_MODEL_WINDOW_TITLE = "Новая модель"
DEELTE_MODEL_BUTTON_TEXT = "Удалить"