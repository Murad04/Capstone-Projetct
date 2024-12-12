import asyncio
import logging
import shelve
import numpy as np
import httpx
#from picamera import PiCamera
import RPi.GPIO as GPIO
#from torchvision import transforms
from logging.handlers import RotatingFileHandler
import faiss
import yolo_setup
import datetime
import os, glob, subprocess
from base_logger import log_function
import os

# Embedding dimension for FAISS
d = 128                                                         # Adjust based on embedding requirements
index = faiss.IndexFlatL2(d)                                    # FAISS index for similarity search

# File paths for cache, image storage, and logs
# File paths for cache, image storage, and logs
CACHE_FILE = "./face_cache/"
IMAGE_DIR = './images/'
LOG_DIR = './logs/'

# Create necessary directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
cache_dir = os.path.dirname(CACHE_FILE)
os.makedirs(cache_dir, exist_ok=True)

# Configure logging with a rotating file handler
handler = RotatingFileHandler(
    LOG_DIR,                                                    # Log directory
    maxBytes=1024 * 1024,                                       # Maximum log size before rotation (1 MB)
    backupCount=2                                               # Number of backup logs to keep
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])
logging.info('Started the code_on_hardware')

# GPIO PIN configuration
START_PIN = 27                                                  # Start button pin
#DOOR_PIN = 18                                                  # Door relay control pin
BEEP_PIN = 18                                                   # Buzzer control pin
SHUT_DOWN_PIN = 22                                              # Shutdown button pin
RESET_PIN = 17                                                  # Reset button pin

# Initialize GPIO pins
GPIO.setmode(GPIO.BCM)                                          # Use Broadcom (BCM) numbering
#GPIO.setup(DOOR_PIN, GPIO.OUT)                                 # Set door pin as output
GPIO.setup(BEEP_PIN, GPIO.OUT)                                  # Set buzzer pin as output
GPIO.setup(START_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)      # Start button with pull-down resistor
GPIO.setup(SHUT_DOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Shutdown button with pull-down resistor
GPIO.setup(RESET_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)      # Reset button with pull-down resistor

# Initialize the camera
#camera = PiCamera()

logging.info("Pushbullet, GPIO setup, picamera done")
logging.info("Starting to load the ML model")

# Load the YOLO model for face detection
yolo_setup.load_model()

#camera.resolution = (640, 480)  

logging.info('Finished loading the model')

# Function to check if a button is pressed
@log_function
def is_btn_pressed(BTN_PIN):
    """Returns True if the specified button pin is pressed."""
    return GPIO.input(BTN_PIN) == GPIO.HIGH

@log_function
async def request_image_from_pc(pc_url):
    """
    Sends a request to the PC to capture an image and saves the image locally.
    Args:
        pc_url (str): The URL of the PC's image capture endpoint.
    Returns:
        str: Path to the saved image on the Raspberry Pi.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{pc_url}/capture_image", timeout=10)
            if response.status_code == 200:
                # Save the image locally
                image_path = f"{IMAGE_DIR}/received_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with open(image_path, "wb") as image_file:
                    image_file.write(response.content)
                logging.info(f"Image received from PC and saved at {image_path}")
                return image_path
            else:
                logging.error(f"Failed to capture image from PC. Status code: {response.status_code}")
                return None
    except httpx.RequestError as e:
        logging.error(f"Request error while capturing image from PC: {e}")
        return None

# Function to capture an image and extract the face
# @log_function
# async def capture_image():
#     """Captures an image using the PiCamera and extracts the face."""
#     # Generate a timestamp for the image file name
#     t_stamp = f'{datetime.date.today()}_{datetime.datetime.now().strftime("%H-%M-%S")}'
#     image_path = f"{IMAGE_DIR}/captured_image_{t_stamp}.jpg"
#     camera.capture(image_path)                                  # Capture the image
#     face_image = yolo_setup.capture_face(image_path)            # Extract the face region
#     return face_image

# Function to check the cache for previous authentication results
@log_function
async def check_cache(embedding, threshold):
    """Searches the cache for a matching embedding within the threshold."""
    D, I = index.search(np.array([embedding]).astype(np.float32), k=1)
    if D[0][0] < threshold:                                     # If the closest match is below the threshold
        with shelve.open(CACHE_FILE) as db:
            return db.get(str(I[0][0]))["result"]               # Return the cached result
    return None

# Function to add embeddings and results to the cache
@log_function
async def add_to_cache(embedding, similarity, result):
    """Adds an embedding and its associated result to the cache."""
    index.add(np.array([embedding]).astype(np.float32))                             # Add embedding to FAISS index
    with shelve.open(CACHE_FILE) as db:
        db[str(index.ntotal - 1)] = {"similarity": similarity, "result": result}    # Store result in cache
    logging.info(f'Added to cache')

# Time interval in seconds to periodically clear the cache
CACHE_CLEAR_INTERVAL = 10000

# Function to clear the cache periodically
@log_function
async def clear_cache_periodically():
    """Clears the cache at regular intervals to free storage space."""
    while True:
        await asyncio.sleep(CACHE_CLEAR_INTERVAL)
        try:
            with shelve.open(CACHE_FILE, writeback=True) as cache:
                cache.clear()
                logging.info("Cache cleared to free storage space on Raspberry Pi.")    
        except Exception as ex:
            logging.error(f'Error clearing cache: {ex}')

# Function to send an image to the cloud for recognition
@log_function
async def send_to_cloud(image_path, cloud_ai_url):
    """Sends an image to a cloud AI service for recognition."""
    url = cloud_ai_url + "/recognize"
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            try:
                response = await client.post(url, files=files, timeout=10)  # 10-second timeout
            except httpx.RequestError as e:
                logging.error(f"Request error: {e}")
                return None
    return response.json()

# Function to control the door and buzzer for access
@log_function
async def process_access():
    """Controls the GPIO to open the door and beep for access."""
    try:
        GPIO.output(BEEP_PIN, GPIO.HIGH)                                    # Activate door relay
        await asyncio.sleep(0.5)                                            # Wait for door to open
        GPIO.output(BEEP_PIN, GPIO.LOW)                                     # Deactivate door relay
        await asyncio.sleep(0.5)                                            # Beep duration
        GPIO.output(BEEP_PIN, GPIO.HIGH)                                     # Deactivate buzzer
    except Exception as ex:
        logging.warning(str(ex))

# Function to clear cached data, images, and logs
@log_function
async def clear_cache():
    """Clears the cache, deletes images, and clears logs."""
    try:
        with shelve.open(CACHE_FILE, writeback=True) as cache:
            cache.clear()
        logging.info('Cache cleared')
    except Exception as ex:
        logging.error(f'Error clearing the cache: {ex}')

    # Delete image files
    for file_path in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
        try:
            os.remove(file_path)
        except Exception as ex:
            logging.error(f'Error deleting image: {file_path}')

    # Delete log files
    for log_file in glob.glob(os.path.join(LOG_DIR, '*.log')):
        try:
            os.remove(log_file)
        except Exception as ex:
            logging.error(f'Error deleting log: {log_file}')

# Function to reset the system and reboot
@log_function
async def reset_system():
    """Clears cache and reboots the Raspberry Pi."""
    await clear_cache()                                                     # Clear cache, images, and logs
    GPIO.cleanup()                                                          # Reset GPIO settings
    logging.info('System reset initiated. Rebooting...')
    subprocess.run(['sudo', 'reboot'])                                      # Reboot the Raspberry Pi

# Main event loop for the system
@log_function
async def main():
    """Main event loop for button handling and system operation."""
    asyncio.create_task(clear_cache_periodically())                         # Start periodic cache clearing
    try:
        while True:
            if is_btn_pressed(START_PIN):                                   # If start button is pressed
                pc_url = "http://<pc-ip>:5000"                              # Replace <pc-ip> with your PC's IP address
                image_path = await request_image_from_pc(pc_url)            # Request an image from the PC
                if image_path:                                              # Proceed if the image was successfully received
                    embedding = await yolo_setup.get_face_embedding(image_path)
                    threshold = 0.5
                    cached_result = await check_cache(embedding, threshold) # Check cache
                    if cached_result and cached_result["result"] == "granted":
                        await process_access()                              # Grant access
                    else:
                        result = await send_to_cloud(image_path, os.getenv('SERVER_MAIN_URL')) # Send to cloud
                        if result and result['result'] == 'granted':
                            await add_to_cache(embedding, result['similarity'], result['result'])
                            await process_access()
                await asyncio.sleep(1)                                      # Delay to prevent button bouncing

            if is_btn_pressed(SHUT_DOWN_PIN):                               # If shutdown button is pressed
                logging.info('Shutdown button pressed')
                break                                                       # Exit main loop to shut down system

            if is_btn_pressed(RESET_PIN):                                   # If reset button is pressed
                logging.info('Reset button pressed')
                await reset_system()                                        # Reset system

            await asyncio.sleep(1)                                          # Poll buttons every second
    except Exception as exp:
        logging.error(str(exp))
    finally:
        GPIO.cleanup()                                                      # Clean up GPIO settings on exit

if __name__ == "__main__":
    asyncio.run(main())                                                     # Start the main event loop