from flask import *
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

#Carregando o modelo
model = load_model('modelo_pet.h5')

#Mapeamento de classes
classes = ['Cat', 'Dog']

@app.route('/home', methods = ['GET', 'POST'])
def home():

    return render_template('home.html')


@app.route("/analisar_imagem", methods=['GET', 'POST'])
def analisarImagem():
    prediction = None
    image_path = None

    arquivo = request.files['imagem']

    try:
        if arquivo:
            filename = secure_filename(arquivo.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            arquivo.save(path)

            # Pré-processa a imagem
            img = load_img(path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predição
            result = model.predict(img_array)
            confidence = np.max(result) * 100  # Pega a maior probabilidade
            predicted_class = classes[np.argmax(result)]

            if confidence >= 80:  # Se a confiança for boa (>80%)
                prediction = f"Após análise, a imagem aparenta ser um animal {predicted_class} (Confiança: {confidence:.2f}%)"
            else:
                prediction = f"Imagem não reconhecida como animal. (Confiança: {confidence:.2f}%)"

            image_path = path

            return jsonify({"prediction": prediction, "image_path": image_path}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500

    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
