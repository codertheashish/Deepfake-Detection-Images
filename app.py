from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
from model.detector import DeepfakeDetector

app = Flask(__name__)
CORS(app)

detector = DeepfakeDetector()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts: JSON with base64 image OR multipart form file
    Returns: { label, confidence, is_fake }
    """
    try:
        if request.content_type and 'multipart' in request.content_type:
            file = request.files.get('image')
            if not file:
                return jsonify({'error': 'No image provided'}), 400
            image = Image.open(file.stream).convert('RGB')

        elif request.is_json:
            data = request.get_json()
            b64 = data.get('image_base64', '')
            if not b64:
                return jsonify({'error': 'No image_base64 field'}), 400
            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        else:
            return jsonify({'error': 'Unsupported content type'}), 415

        result = detector.predict(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': detector.model_name})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
