from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
app.secret_key = '950267365js@'  # Change this to a strong, unique key

# Specify the directory where you want to save uploaded files
upload_dir = 'C:\\Users\\yelim\\Downloads\\4\\uploads'
os.makedirs(upload_dir, exist_ok=True)

# Specify the allowed file extensions
allowed_extensions = {'jpg', 'jpeg', 'png'}

# Load your trained dermatology model
model = load_model('C:\\Users\\yelim\\Downloads\\save\\dermatology.h5')

# Mapping between class labels and disease names
class_to_disease = {
    0: "Melanoma",
    1: "Melanocytic Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Dermatofibroma",
    6: "Vascular Lesion",
    7: "Squamous Cell Carcinoma",
    8: "No Skin Diseases Detected"
}

# Define the probability threshold below which "None of the Above" will be displayed
probability_threshold = 0.65

# Function to check if a filename has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis_result = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Check if the file has a valid filename
            if file.filename == '':
                flash('No selected file')
            elif not allowed_file(file.filename):
                flash('Invalid file extension')
            else:
                # Save the uploaded file to the upload directory
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_dir, filename))

                # Load the saved image using Keras
                image_path = os.path.join(upload_dir, filename)
                img = image.load_img(image_path, target_size=(224, 224))

                # Preprocess the image (you may need to adapt this preprocessing to your model)
                img = image.img_to_array(img)
                img = tf.keras.applications.resnet50.preprocess_input(img)
                img = tf.expand_dims(img, axis=0)  # Add batch dimension

                # Perform inference with your model
                diagnosis = model.predict(img)[0]

                # Find the top predicted class index
                top_class_index = np.argmax(diagnosis)

                # Check if the top predicted class is "None of the Above"
                if top_class_index == 8:  # 8 corresponds to "None of the Above" in class_to_disease
                    diagnosis_result = "No Skin Diseases Detected"
                else:
                    # Get the probability of the top predicted disease
                    top_probability = diagnosis[top_class_index]

                    # Check if the probability is below the threshold
                    if top_probability < probability_threshold:
                        diagnosis_result = "No Skin Diseases Detected"
                    else:
                        # Get the corresponding disease name and probability for the top predicted disease
                        top_disease = class_to_disease[top_class_index]

                        # Construct the diagnosis result
                        diagnosis_result = f"{top_disease}: {top_probability:.2f}"

                # Cleanup: You can optionally delete the uploaded file after processing
                os.remove(image_path)

                # Redirect to the result page with diagnosis result as a query parameter
                return redirect(url_for('result', diagnosis_result=diagnosis_result))

    return render_template('index.html', diagnosis_result=diagnosis_result)

@app.route('/result')
def result():
    diagnosis_result = request.args.get('diagnosis_result', '')
    return render_template('result.html', diagnosis_result=diagnosis_result)

if __name__ == '__main__':
    app.run(debug=True)




