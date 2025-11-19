import flask
from flask import request, jsonify, render_template
from deepface import DeepFace
import os
import numpy as np
import base64
import io
from PIL import Image
import csv  # <-- Import CSV library
import datetime  # <-- Import datetime library

app = flask.Flask(__name__)

# --- Setup ---
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
RECOGNITION_THRESHOLD = 0.40  # Adjust as needed

# Initialize the attendance file with headers if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

try:
    print("Warming up DeepFace model...")
    DeepFace.find(
        img_path=np.zeros((100, 100, 3), dtype=np.uint8),
        db_path=KNOWN_FACES_DIR,
        model_name="VGG-Face",
        enforce_detection=False
    )
    print("DeepFace model is ready.")
except Exception as e:
    print(f"Error during DeepFace warmup: {e}")


def decode_image_from_base64(base64_string):
    if "," in base64_string:
        header, encoded_data = base64_string.split(',', 1)
    else:
        encoded_data = base64_string
    image_data = base64.b64decode(encoded_data)
    image = Image.open(io.BytesIO(image_data))
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = image_np[..., :3]
    image_np_bgr = image_np[:, :, ::-1]
    return image_np_bgr


def check_and_log_attendance(name):
    """
    Checks if a user is already marked present today.
    If not, logs them and returns 'success'.
    If yes, returns 'already_checked_in'.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # Check for duplicates
    with open(ATTENDANCE_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == name and row[1] == today:
                return "already_checked_in"  # Already logged today

    # If no duplicate, log the new attendance
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, today, current_time])
        return "success"


# --- Web Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data_base64 = data['image']
    
    try:
        unknown_image = decode_image_from_base64(image_data_base64)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({'error': 'Could not decode image'}), 400

    # --- 1. Emotion Analysis (Wrapped in Safety Block) ---
    dominant_emotion = "neutral" 
    try:
        analysis = DeepFace.analyze(
            img_path=unknown_image, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv' 
        )
        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            dominant_emotion = analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"⚠️ Emotion analysis skipped: {e}")
        dominant_emotion = "neutral"

    # --- 2. Face Recognition ---
    recognized_names = []
    try:
        result_dfs = DeepFace.find(
            img_path=unknown_image,
            db_path=KNOWN_FACES_DIR,
            model_name="VGG-Face",
            distance_metric="cosine", # Explicitly request cosine
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        for df in result_dfs:
            if not df.empty:
                best_match = df.iloc[0]
                
                # --- ✨ BUG FIX: Flexible Distance Check ---
                # DeepFace sometimes changes column names. We check for all possibilities.
                distance = 100 # Default to "far away"
                
                if 'distance' in best_match:
                    distance = best_match['distance']
                elif 'VGG-Face_cosine' in best_match:
                    distance = best_match['VGG-Face_cosine']
                elif 'cosine' in best_match:
                    distance = best_match['cosine']
                
                # Only add if the match is close enough
                if distance <= RECOGNITION_THRESHOLD:
                    identity_path = best_match['identity']
                    filename = os.path.basename(identity_path)
                    name = os.path.splitext(filename)[0]
                    recognized_names.append(name)
                    
    except Exception as e:
        print(f"Error during DeepFace find: {e}")
        # Return empty lists so the frontend doesn't show a red error
        return jsonify({'new_check_ins': [], 'already_present': [], 'emotion': 'neutral'})

    # --- 3. Logging Logic ---
    unique_names = list(set(recognized_names))
    new_check_ins = []
    already_present = []

    for name in unique_names:
        status = check_and_log_attendance(name)
        if status == "success":
            new_check_ins.append(name)
        elif status == "already_checked_in":
            already_present.append(name)

    return jsonify({
        'new_check_ins': new_check_ins,
        'already_present': already_present,
        'emotion': dominant_emotion
    })


# --- Main ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)