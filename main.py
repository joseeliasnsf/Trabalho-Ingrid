import tensorflow as tf
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Função para carregar a imagem a partir de uma URL
def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Não foi possível carregar a imagem a partir da URL: {url}")
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    return img

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(img_url):
    img = load_image_from_url(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte de BGR para RGB
    img = cv2.resize(img, (224, 224))  # Redimensiona para 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

# Função para fazer a previsão
def classify_image(model, img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# URL da imagem
img_url = "https://th.bing.com/th/id/OIP.J6A9scj1ErA0Lt1GfnJZ2QHaEK?w=298&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7"

# Carregar e pré-processar a imagem
try:
    original_img, processed_img = load_and_preprocess_image(img_url)
except ValueError as e:
    print(e)
    exit()
    
# Carregar o modelo pré-treinado MobileNetV2
model = MobileNetV2(weights='imagenet')

# Classificar a imagem
predictions = classify_image(model, processed_img)

# Exibir a imagem e os resultados da classificação
plt.imshow(original_img)
plt.axis('off')
plt.title("Resultados da Classificação")
plt.show()

for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i+1}: {label} ({score*100:.2f}%)")