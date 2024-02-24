import logging
from flask import Flask, request, jsonify, send_file, Response
import time
from flask import url_for
import os
import opslib
import cv2
import io
from PIL import Image
import base64
import numpy as np
import torch
from model.model import LSTMModel
# from vis.processor import Processor
from default_params import *
from flask import jsonify
from helpers import pop_and_add, last_ip, dist, move_figure, get_hist
# Set path for the model
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
@app.route('/get_image',methods=['GET'])
def get_image():
    while True:
        # Replace this with your image processing logic
        # For example, read an image file from disk
        image_path = 'path_to_your_image.jpg'

        # Send the image file in chunks
        try:
            return send_file(image_path, mimetype='image/jpeg')
        except Exception as e:
            print(e)
            time.sleep(1)  # Wait for 1 second before retrying

@app.route("/predict", methods=['POST'])
   
def predict():
    json_payload = request.json
    print(json_payload)
    instance = json_payload.get('instance', None)  # Assuming instance contains base64 encoded image data
    if instance is None:
        return jsonify({"error": "No instance data found"}), 400
    
    # Decode base64 image data
    try:
        image_data = base64.b64decode(instance)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    image_array = np.array(image)
    app.logger.info(f"json_payload: {instance}")
    
    # prediction = opslib.predict(image_array)
    # app.logger.info(f"predictions: {prediction}")
    # predictions_ref = db.reference('predictions')
    # predictions_ref.push({'frame': instance ,'isFall': prediction})
    return {'result':'Upload to firebase successfully'}
@app.route('/')
def index():
    url = url_for('index', _external=True)
    print(f"Server running at {url}")
    return 'Hello, World!'
VIDEO_FILE_PATH = "downloaded_video.mp4"
@app.route('/video_feed')
def video_feed():
    #return Response(opslib.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(opslib.generate_frames(VIDEO_FILE_PATH,30), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video')
def get_video():
    opslib.get_webcam()
    return "hello"
    # return send_file(VIDEO_FILE_PATH, mimetype='video/mp4')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)