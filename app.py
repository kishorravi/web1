# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the image data from the request
    img_data = request.form['image']
    
    # Convert base64 image data to NumPy array
    img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
    
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
