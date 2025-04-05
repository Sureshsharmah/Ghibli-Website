import os
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import uuid
from flask_cors import CORS
import base64
import onnxruntime
from PIL import Image, ImageEnhance
import os


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['MODEL_PATH'] = os.path.join("model", "AnimeGANv2_Hayao.onnx")

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize ONNX runtime session
try:
    ort_session = onnxruntime.InferenceSession(
        app.config['MODEL_PATH'],
        providers=['CPUExecutionProvider']
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    ort_session = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((512, 512))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def postprocess_image(output):
    try:
        output = output.squeeze(0)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to PIL for further enhancement
        output_img = Image.fromarray(output)
        
        # Improve sharpness and contrast for a cleaner cartoon effect
        enhancer = ImageEnhance.Sharpness(output_img)
        output_img = enhancer.enhance(2.0)  # Increase sharpness
        enhancer = ImageEnhance.Contrast(output_img)
        output_img = enhancer.enhance(1.5)  # Increase contrast
        
        return np.array(output_img)
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        raise

def apply_ghibli_style(input_path, output_path):
    try:
        if ort_session is None:
            raise ValueError("Model not loaded")
        input_img = preprocess_image(input_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        output = ort_session.run([output_name], {input_name: input_img})[0]
        result_img = postprocess_image(output)
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Error in style transfer: {e}")
        return False

@app.route('/')
def index():
    permanent_image = "Untitled_profile .jpg"
    additional_image = "profile1.jpg"

    permanent_image_path = os.path.join(app.config['RESULT_FOLDER'], permanent_image)
    additional_image_path = os.path.join(app.config['RESULT_FOLDER'], additional_image)

    permanent_image_url = url_for('static', filename=f'results/{permanent_image}') if os.path.exists(permanent_image_path) else None
    additional_image_url = url_for('static', filename=f'results/{additional_image}') if os.path.exists(additional_image_path) else None

    return render_template(
        'index.html', 
        permanent_image_url=permanent_image_url, 
        has_permanent_image=permanent_image_url is not None,
        additional_image_url=additional_image_url,
        has_additional_image=additional_image_url is not None
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed', 'status': 'error'}), 400
    
    try:
        # Ensure upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Ensure results folder exists
        if not os.path.exists(app.config['RESULT_FOLDER']):
            os.makedirs(app.config['RESULT_FOLDER'])
        
        filename = f"{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        output_filename = f"ghibli_{filename}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        success = apply_ghibli_style(input_path, output_path)
        if not success:
            return jsonify({'error': 'Failed to process image', 'status': 'error'}), 500
        
        # Read the result image
        with open(output_path, 'rb') as f:
            result_data = f.read()
        result_base64 = base64.b64encode(result_data).decode('utf-8')
        
        # Return proper JSON response
        response = jsonify({
            'status': 'success',
            'original': f"/static/uploads/{filename}",
            'result': f"/static/results/{output_filename}",
            'preview': f"data:image/jpeg;base64,{result_base64}"
        })
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({
            'error': f'Failed to process image: {str(e)}', 
            'status': 'error'
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
