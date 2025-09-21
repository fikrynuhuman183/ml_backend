from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from dotenv import load_dotenv
import util


app = Flask(__name__)
load_dotenv()
CORS(app)

img_height = 224
img_width = 224

@app.route('/hello')
def hello():
    return "Hello, World!"

@app.route('/classify', methods=['POST'])
def classify():
    try:
        filename = request.json['filename']
        supported_extensions = ['.jpg', '.jpeg', '.png', '']
        
        image_path = None
        
        for ext in supported_extensions:
            path = os.path.join(os.getenv('IMAGES_PATH'), filename+ext)
            if os.path.isfile(path):
                image_path = path
                break
        
        if image_path is None:
            return jsonify({'error': 'Image not found'}), 404
        
        # Use only H5 model prediction
        pred = util.nnPredict(image_path)
        finalPred = pred[0]  # Get the first (and only) prediction

        return jsonify({
            'result': {
                'benign': str(finalPred[0]),
                'healthy': str(finalPred[1]),
                'opmd': str(finalPred[2])
            }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))