import base64
import cv2
from tensorflow.keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image
from io import BytesIO


def preprocess_image(image_bytes):
    image_bytes = image_bytes.resize((28, 28))
    image_bytes = image_bytes.convert('L')
    image_bytes = image.img_to_array(image_bytes)
    image_bytes = image_bytes.reshape(784)  # Меняем форму массива в плоский вектор
    image_bytes = 255 - image_bytes  # Инвертируем изображение
    image_bytes /= 255  # Нормализуем изображение

    return image_bytes


def preprocess_path_to_file(image):
    # Изменение размера изображения до 28x28пикселей
    resized = cv2.resize(image, (28, 28))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Нормализация значений пикселей
    normalized = gray / 255.0

    # Преобразование изображения в массив NumPy
    array = np.array(normalized)

    # Добавление размерности для совместимости с моделью машинного обучения
    processed = np.expand_dims(array, axis=0)  # Нормализуем изображение

    return processed


def get_result(image_file, is_api=False):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
               'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    model = models.load_model('model/emnist.h5')
    try:
        image_bytes = image_file.file.read()
        encoded_string = base64.b64encode(image_bytes)
        bs64 = encoded_string.decode('utf-8')
        image_data = f'data:image/jpeg;base64,{bs64}'

        img = Image.open(BytesIO(image_bytes))
        img = preprocess_image(img)
        x = img
        x = np.expand_dims(x, axis=0)
        x = x.reshape(-1, 28, 28, 1)
        prediction = model.predict(x)
    except ValueError:
        img = preprocess_path_to_file(image_file)
        prediction = model.predict(img)
    except AttributeError:
        img = preprocess_path_to_file(image_file)
        prediction = model.predict(img)

    index = np.argmax(prediction)  # находит индекс максимального элемента

    result = {
        "inference_time": str(round(prediction[0, index] * 100, 2)),
        "predictions": {
            "class_id": str(index),
            "class_name": str(classes[index])
        }
    }

    if not is_api:
        result["image_data"] = image_data

    return result
