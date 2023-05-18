from fastapi.testclient import TestClient
import main
from utils import get_result
import numpy as np
import cv2

client = TestClient(main.app)


def test_read_main():  # проверяет доступность приложения при обращении к корню сервер
    response = client.get("/")
    assert response.status_code == 200


def test_predict():  #
    file_name = 'static/images/test_image.jpg'
    response = client.post("/predict", files={"file": ("test_image", open(file_name, "rb"), "image/jpeg")})
    assert response.status_code == 200
def test_get_result():
    # загружаем тестовое изображение
    file_name = 'static/images/test_image.jpg'
    image = cv2.imread(file_name)

    result = get_result(image, is_api=True)


    # проверяем, что ответ содержит ожидаемые поля
    assert "inference_time" in result
    assert "predictions" in result
    assert "class_id" in result["predictions"]
    assert "class_name" in result["predictions"]

    # проверяем, что предсказание верно
    assert result["predictions"]["class_name"] == "q"
