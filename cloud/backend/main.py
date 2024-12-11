from quart import Quart, request, jsonify, render_template, websocket
import torch
import cv2
import numpy as np
import backend_db_operations as  b_DB
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from pushbullet import Pushbullet
from base_logger import log_function
from quart_cors import cors
import httpx
import sys
sys.path.insert(0,r'D:\\Personal\\codes\\project capstone\\cloud\\ml_yolov5')
import model_yolo_for_recog as M_Yolo_Recog
import asyncio,datetime,aiofiles
import traceback, time,json,re
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pushbullet API for sending notifications
pb = Pushbullet(os.getenv("PUSHBULLET_API_KEY"))
logging.info("Started the project")

# Initialize Quart app and enable Cross-Origin Resource Sharing (CORS)
app = Quart(__name__, template_folder='../frontend')
app = cors(app, allow_origin='*')
app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True
logging.info('Started preprocessing features')

# Preprocessing pipeline for face images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),      # Resize face image to 112x112
    transforms.ToTensor(),              # Convert image to tensor
])

# Load the YOLO model for face detection
logging.info("Loading the model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = M_Yolo_Recog.load_model(device)
M_Yolo_Recog.load_model_once()
device = M_Yolo_Recog.get_device()  # Get the device (CPU/GPU) the model is using

# Load or initialize the known faces database
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}                    # Initialize an empty database if the file doesn't exist

@log_function
# Function to wait for a "yes" response from Pushbullet
async def wait_for_response():
    start_time = time.time()
    timeout = 120  # 2 minutes in seconds

    print("Waiting for 'yes' response...")
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logging.info("Timeout reached. No response received.")
            return 'not granted'
        
        # Fetch the latest pushes
        pushes = pb.get_pushes(limit=1)  # Limit to last 10 pushes to reduce load
        for push in pushes:
            # Check if the push contains the text "yes"
            if push.get("body", "").strip().lower() == "yes":
                logging.info("Received 'yes' response!")
                return 'granted'
            else:
                return 'not granted'

        # Add a short delay to avoid spamming the API
        time.sleep(5)


# Save the known faces database to a file
@log_function
async def save_known_faces():
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# Compare face embeddings to recognize the person
@log_function
async def recognize_face(embedding):
    name = "Unknown"
    max_similarity = 0.75
    threshold = 0.6                     # Similarity threshold
    for person, known_embedding in known_faces.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > threshold and similarity > max_similarity:
            max_similarity = similarity
            name = person
    return name, max_similarity

# Serve the home page
@app.route("/")
@log_function
async def index():
    try:
        return await render_template('/home/index.html')
    except Exception as ex:
        logging.error(f'Error: {ex}')

# Endpoint to fetch usernames from the database
@app.route('/get_names', methods=['GET'])
@log_function
async def get_names():
    table_name = 'Users'
    columns_to_fetch = ['username']
    try:
        names = await b_DB.get_custom_data_from_custom_table(table_name, columns_to_fetch)
        result = [dict(row) for row in names]
        return jsonify({"names": result})
    except Exception as ex:
        logging.error(f"Error at get_names: {ex}")
        return jsonify({"error": "Failed to fetch the data"}), 500

# Endpoint to fetch user login logs
@app.route("/get_user_login_logs", methods=['GET'])
@log_function
async def get_user_login_logs():
    try:
        data = await b_DB.get_all_visitor_logs_with_usernames()
        return jsonify(data)      
    except Exception as ex:
        logging.error(f'Error at getting the user login logs: {ex}')
        return jsonify({"error": "Failed to fetch the data"}), 500
    
@log_function
@app.route("/recognize", methods=["POST"])
async def recognize():
    try:
        # Validate and read the uploaded file
        files = await request.files
        if "file" not in files:
            return jsonify({"error": "No file part in the request."}), 400

        file = files["file"]
        file_data = file.read()
        if not file_data:
            return jsonify({"error": "The file is empty."}), 400

        # Decode the image
        np_img = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        # Save a temporary file
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, img)

        # Perform detection
        detections = M_Yolo_Recog.call_ML(temp_image_path)
        logging.info(f"Detections: {detections}")

        # Process detections
        if detections and len(detections) > 0 and 'no detections' not in str(detections):
            logging.info("Access granted")
            return jsonify({'result': 'granted'})

        # If no detections, send notification
        main_url = 'http://192.168.45.105:5000'
        notification_payload = {
            "title": "Face Recognition Alert",
            "message": "No face detected. Do you want to grant access?",
            "user_id": 123,  # Replace with the actual user ID if applicable
            "log_id": 456,   # Replace with the actual log ID if applicable
            "notification_type": "access_request",
            'file': r"D:\\Personal\\codes\\project capstone\\temp_image.jpg"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Send the image file and payload to the notification endpoint
                notify_response = await client.post(
                    f"{main_url}/send_notification",
                    data=notification_payload,  # Other data fields  
                )
                logging.info(f"Notification Response Status: {notify_response.status_code}")
                logging.info(f"Notification Response Body: {notify_response.text}")

                if notify_response.status_code == 200:
                    response_data = notify_response.json()
                    if response_data.get("result") == "granted":
                        logging.info("Access granted by user")
                        return jsonify({'result': 'granted'})
                    else:
                        logging.info("Access denied by user")
                        return jsonify({'result': 'denied'}), 403
                else:
                    logging.error(f"Failed to send notification: {notify_response.status_code}")
                    return jsonify({"error": "Failed to send notification"}), 500
            except Exception as ex:
                error_traceback = traceback.format_exc()
                logging.error(f"Error sending notification: {error_traceback}")
                return jsonify({
                    'error': str(ex),
                    'traceback': error_traceback
                }), 500

    except Exception as ex:
        # Capture and return traceback for debugging
        error_traceback = traceback.format_exc()
        logging.error(f"Error in recognize endpoint: {error_traceback}")
        return jsonify({
            "error": str(ex),
            "traceback": error_traceback
        }), 500


# Endpoint to add a new user to the database
@app.route('/add_user',methods = ['POST'])
@log_function
async def add_user():
    existing_user = await b_DB.get_custom_data_from_custom_table(
        "Users", ["user_id"], {"email": email}
    )
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    name =          request.form['name']
    email =         request.form['email']
    password =      request.form['password']
    role =          request.form['role']
    created_at =    datetime.datetime.now()
    last_login =    created_at
    data = {
        'name':         name,
        'email':        email,
        'password':     password,
        'role':         role,
        'created_at':   created_at,
        'last_login':   last_login
    }
    await b_DB.add_data('Users',data)

# Endpoint to add a new face to the database
@app.route("/add_face", methods=["POST"])
@log_function
async def add_face():
    if "file" not in request.files or "name" not in request.form:
        return jsonify({"error": "File and name are required"}), 400
    name =      request.form["name"]
    email =     request.form['email']
    password =  request.form['password']
    
    # Validate user credentials
    user_id = await b_DB.get_userID(name, email, password)
    if not user_id:
        return jsonify({"error": "User not found or invalid credentials"}), 404
    
    # Process and save face embedding
    file =          await request.files["file"].read()
    np_img =        np.frombuffer(file, np.uint8)
    img =           cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_tensor =    preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = M_Yolo_Recog.extract_face_embeddings(img_tensor).cpu().numpy().flatten()
    
    # Save image and embedding
    image_path = f"faces/{name}_{datetime.datetime.now().timestamp()}.jpg"
    cv2.imwrite(image_path, img)
    
    data_for_face = {
        'user_id':       user_id,
        'name':          name,
        'image_path':    image_path,
        'encoding_data': pickle.dumps(embedding),
        'added_at':      datetime.datetime.now(),
        'is_active':     True,
    }
    
    try:
        await b_DB.add_data('Faces', data_for_face)
        known_faces[name] = embedding           # Update known_faces in memory
        await save_known_faces()                # Persist known_faces to file
        return jsonify({"message": f"Face for {name} added successfully."})
    except Exception as ex:
        logging.error(f"Failed to add face: {ex}")
        return jsonify({"error": "Failed to add face to database"}), 500

# Endpoint to send notifications via Pushbullet
@app.route('/send_notification', methods=['POST'])
@log_function
async def send_notification():
    data =                  await request.form
    logging.info(f'data reeceied in notify: {data}')
    title =                 data.get('title')
    message =               data.get('message')
    user_id =               data.get('user_id')
    log_id =                data.get('log_id')
    notification_type =     data.get('notification_type')
    sent_at =               datetime.datetime.now()
    file_path =             data.get('file')             
    
    
    with open(file_path, "rb") as pic:
        file_data = pb.upload_file(pic, "picture.jpg")

    # Send notification via Pushbullet
    try:
        pb.push_file(**file_data)
        pb.push_note(title, message)
        response = await wait_for_response()
        # Log notification in the database
        await b_DB.store_notifications(notification_type, sent_at)
        logging.info({"message": "Notification sent successfully"})
        if response == 'granted':
            logging.info('Granted by user')
            return jsonify({'result':'granted'})
        else:
            logging.info("Denied by user")
            return jsonify({'result':'not granted'})
    except Exception as ex:
        error_traceback = traceback.format_exc()
        logging.error(f"Error in send notification endpoint: {error_traceback}")
        return jsonify({
            "error": str(ex),
            "traceback": error_traceback
        }), 500

# Endpoint for user login
@app.route("/login", methods=['POST'])
@log_function
async def login():
    try:
        data =          await request.get_json()
        email =         data.get('email')
        password =      data.get('password')
        username =      data.get('username')
        user_id =       await b_DB.get_userID(username, email, password)
        result_admin =  await b_DB.login_to_admin(password, user_id, email)
        if result_admin:
            return jsonify({"message": "Login successful", "redirect_url": "/admin"}), 200
        return jsonify({"error": "Unauthorized login"}), 403
    except Exception as ex:
        return jsonify({"error": f"{ex}"}), 500
    
# Endpoint for login page
@log_function
async def login_page():
    return await render_template('/login/login.html')

# WebSocket endpoint for real-time updates
@log_function
@app.websocket('/ws')
async def websocket_endpoint():
    try:
        while True:
            notifications  = await b_DB.get_notifications_by_user()
            if notifications:
                await websocket.send_json({"notifications":notifications})
            await asyncio.sleep(2)
    except Exception as ex:
        logging.error(f'Error in websocket endpoint: {ex}')

# Admin and user management endpoints
@app.route("/admin")
@log_function
async def admin_page():
    return await render_template("admin.html")

@app.route('/add_user_page')
@log_function
async def add_user_page():
    return await render_template("add_user.html")

# Run the Quart app
if __name__ == "__main__":
#def run():
    app.run(host="0.0.0.0", port=5000)