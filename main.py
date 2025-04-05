import os
import uuid
import base64
import cv2
import numpy as np
import tempfile
import logging
import gc  # Garbage collection
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import onnxruntime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Config
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # Reduced to 4MB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['MODEL_PATH'] = os.path.join(BASE_DIR, 'model', 'AnimeGANv2_Hayao.onnx')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'image')

permanent_image = "profile2.jpg"
additional_image = "profile1.jpg"

# Global variable to control whether we actually use the model
USE_MODEL = os.environ.get('USE_MODEL', 'true').lower() == 'true'

# Don't load the model yet - we'll load it only when needed
ort_session = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_if_needed():
    global ort_session
    if ort_session is not None:
        return True
        
    if not USE_MODEL:
        logger.info("Model usage is disabled by environment variable")
        return False
        
    try:
        model_path = app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist at: {model_path}")
            return False
            
        # Use minimal resources
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        logger.info(f"Loading model from: {model_path}")
        ort_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image_path):
    try:
        # Open and drastically reduce image size to save memory
        img = Image.open(image_path).convert("RGB")
        
        # Always resize to a small size to save memory
        img = img.resize((256, 256), Image.LANCZOS)
        
        # Convert to numpy array with float32 precision
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Clear references to help garbage collection
        img = None
        gc.collect()
        
        return img_array
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def postprocess_image(output):
    try:
        # Process the model output
        output = output.squeeze(0)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Convert to uint8 for PIL
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Apply minimal enhancements
        output_img = Image.fromarray(output)
        output_img = ImageEnhance.Contrast(output_img).enhance(1.2)
        
        # Convert back to numpy array
        result = np.array(output_img)
        
        # Clear references
        output_img = None
        gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Error in postprocessing: {str(e)}")
        raise

def apply_ghibli_style(input_path):
    try:
        # Load model if needed
        if not load_model_if_needed():
            raise RuntimeError("Model not available")

        # Process the image
        input_img = preprocess_image(input_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Run inference
        logger.info("Running inference")
        output = ort_session.run([output_name], {input_name: input_img})[0]
        
        # Clear input to save memory
        input_img = None
        gc.collect()
        
        # Process the output
        result_img = postprocess_image(output)
        
        # Encode as JPEG with lower quality to save memory
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR), 
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Clear result
        result_img = None
        gc.collect()
        
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in style transfer: {str(e)}")
        raise

def image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None
            
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        return None

@app.route('/')
def index():
    try:
        # Build absolute paths to images
        permanent_path = os.path.join(IMAGE_FOLDER, permanent_image)
        additional_path = os.path.join(IMAGE_FOLDER, additional_image)
        
        # Convert images to base64
        permanent_b64 = image_to_base64(permanent_path)
        additional_b64 = image_to_base64(additional_path)

        return render_template(
            'index.html',
            permanent_image_url=f"data:image/jpeg;base64,{permanent_b64}" if permanent_b64 else None,
            has_permanent_image=permanent_b64 is not None,
            additional_image_url=f"data:image/jpeg;base64,{additional_b64}" if additional_b64 else None,
            has_additional_image=additional_b64 is not None
        )
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    # Handle favicon requests to prevent errors
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Upload request received")
        
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        file.save(temp_path)
        logger.info(f"File saved to temporary path: {temp_path}")

        # Read the original file for base64 encoding (reduce quality)
        try:
            img = Image.open(temp_path)
            img = img.resize((400, int(400 * img.height / img.width)), Image.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as small_file:
                small_path = small_file.name
                img.save(small_path, quality=85)
                with open(small_path, 'rb') as f:
                    original_b64 = base64.b64encode(f.read()).decode('utf-8')
                os.unlink(small_path)
        except Exception as e:
            logger.error(f"Error processing original image: {str(e)}")
            with open(temp_path, 'rb') as f:
                original_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Process image
        try:
            result_b64 = apply_ghibli_style(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass

            return jsonify({
                'status': 'success',
                'original': f"data:image/jpeg;base64,{original_b64}",
                'result': f"data:image/jpeg;base64,{result_b64}"
            })
        except Exception as processing_error:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
            logger.error(f"Image processing error: {str(processing_error)}")
            return jsonify({
                'status': 'error', 
                'message': f'Image processing failed: {str(processing_error)}'
            }), 500

    except Exception as e:
        logger.error(f"Upload route error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
