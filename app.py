from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_furniture', methods=['POST'])
def add_furniture():
    # Get the image data and furniture position from the request
    img_data = request.form['image']
    furniture_x = int(request.form['furnitureX'])
    furniture_y = int(request.form['furnitureY'])

    # Convert base64 image data to NumPy array
    img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)

    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Add a simple rectangle representing a sofa to the image
    cv2.rectangle(img, (furniture_x, furniture_y), (furniture_x + 100, furniture_y + 50), (0, 0, 255), 2)

    # Encode the modified image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
