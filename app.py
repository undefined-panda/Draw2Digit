from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
from PIL import Image
from backend.digit_recognition import model_prediction

app = Flask(__name__, static_folder='frontend/assets', template_folder='frontend')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pages/digit_recognition')
def digit_recognition():
    return render_template('pages/digit_recognition.html')

@app.route('/pages/digit_recognition', methods=["POST"])
def upload_image():
    data = request.json
    image_data = data['drawing']
    image = np.array(image_data, dtype=np.uint8)

    image = Image.fromarray(image)
    new_size = (28,28)

    # resizing the image
    resized_image = image.resize(new_size)
    resized_image = np.array(resized_image, dtype=np.uint8)

    cnn_prediction = model_prediction(resized_image)

    converted_image = Image.fromarray(resized_image)
    img_io = io.BytesIO()
    converted_image.save(img_io, "PNG")
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    response = {
        "status": "success",
        "message": "Daten erfolgreich verarbeitet",
        "processed_value": cnn_prediction,
        "image": img_base64
    }

    return jsonify(response), 200 

if __name__ == '__main__':
    app.run()