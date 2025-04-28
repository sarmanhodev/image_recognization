from flask import *
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

#Carregando o modelo
model = load_model('modelo_jogadores.h5')
# Carrega o modelo base MobileNetV2 (sem a parte final de classificação)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

#Mapeamento de classes
classes = ['Cat', 'Dog']

# Função para extrair características de uma imagem
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    return features.flatten()  # Retorna um vetor 1D

@app.route('/home', methods = ['GET', 'POST'])
def home():

    return render_template('home.html')


@app.route("/analisar_imagem", methods=['POST'])
def analisarImagem():
    # Variáveis para retorno
    prediction = None
    image_path = None
    max_similarity = -1
    best_match = None
    similarity_threshold = 0.8  # Defina o limite de similaridade mínima

    # Recebe o arquivo de imagem
    arquivo = request.files['imagem']
    print(arquivo)

    try:
        if arquivo:
            filename = secure_filename(arquivo.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            arquivo.save(path)

            # Extração das características da imagem do upload
            upload_features = extract_features(path)

            # Itera sobre todas as imagens de treinamento e compara
            for filename_train in os.listdir("jogadores_dataset/"):
                if filename_train.endswith(('.png', '.jpg', '.jpeg')):
                    # Caminho da imagem do treinamento
                    train_path = os.path.join("jogadores_dataset/", filename_train)

                    # Extrai as características da imagem de treinamento
                    train_features = extract_features(train_path)

                    # Calcula a similaridade de cosseno entre as características
                    similarity = cosine_similarity([upload_features], [train_features])[0][0]

                    # Atualiza o melhor match
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = filename_train

            # Verifica se a similaridade atingiu o limite mínimo
            if max_similarity >= similarity_threshold:
                # Extrai o nome do jogador da imagem mais similar
                player_name = best_match.split('_')[0].capitalize()  # Exemplo: 'messi_1.jpg' -> 'messi'
                prediction = f"Após análise, a imagem aparenta ser o {player_name} (Similaridade: {max_similarity:.2f})"
            else:
                prediction = f"Imagem não reconhecida. Similaridade muito baixa. (Similaridade: {max_similarity:.2f})"

            image_path = path

            return jsonify({"prediction": prediction, "image_path": image_path}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
