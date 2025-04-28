import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import os

# Caminho do dataset
dataset_path = os.path.join(os.getcwd(), "clubes_dataset", "times")

# Pré-processamento sem validação (pois temos poucas imagens)
datagen = ImageDataGenerator(rescale=1./255)

# Parâmetros
img_size = (224, 224)
batch_size = 2  # Pequeno porque temos poucas imagens

# Gerador
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Carregando a base MobileNetV2 pré-treinada
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Modelo final
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes
])

# Compilar
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar
history = model.fit(
    train_generator,
    epochs=10
)

# Salvar
model.save("modelo_clubes.h5")
print("\n✅ Modelo salvo como 'modelo_clubes.h5'\n")
