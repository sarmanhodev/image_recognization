import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

# Caminho do dataset
dataset_path = "jogadores_dataset/"  # Caminho para a pasta contendo as imagens dos jogadores

# Criar um DataFrame com os nomes dos arquivos e suas respectivas classes (nome do jogador)
data = []

# Iterar sobre os arquivos de imagem e criar uma lista para o DataFrame
for filename in os.listdir(dataset_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Considerando imagens PNG, JPG, e JPEG
        # O nome do jogador será extraído do nome do arquivo antes do primeiro "_", por exemplo: 'messi_1.png' -> 'messi'
        jogador = filename.split('_')[0]  # Usando a parte do nome do arquivo antes do primeiro "_"
        data.append([os.path.join(dataset_path, filename), jogador])

# Criar o DataFrame com caminho da imagem e classe (nome do jogador)
df = pd.DataFrame(data, columns=["filename", "class"])

# Pré-processamento e aumento de dados
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Separando 20% para validação
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Parâmetros
img_size = (224, 224)  # MobileNetV2 trabalha melhor com 224x224
batch_size = 32

# Geradores com flow_from_dataframe
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"  # Usando 80% dos dados para treinamento
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"  # Usando 20% dos dados para validação
)

# Carregando a base MobileNetV2 pré-treinada
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar para usar como extrator de características

# Montar o modelo final
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(df["class"].unique()), activation="softmax")  # Número de classes (jogadores)
])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Treinamento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25
)

# Salvar modelo treinado
model.save("modelo_jogadores.h5")

print("\n✅ Modelo salvo como 'modelo_jogadores.h5'\n")
