import asyncio
import logging
import RPi.GPIO as GPIO
import os
import datetime
import subprocess
from logging.handlers import RotatingFileHandler

# GPIO PIN configuration
START_PIN = 27  # Start button pin
SHUT_DOWN_PIN = 22  # Shutdown button pin
RESET_PIN = 17  # Reset button pin
BEEP_PIN = 18  # Buzzer pin

# Logging configuration
LOG_DIR = './logs/'
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'app.log')

handler = RotatingFileHandler(
    log_file_path, maxBytes=1024 * 1024, backupCount=2
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(START_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SHUT_DOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(RESET_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BEEP_PIN, GPIO.OUT)

# Button Callbacks
def start_button_callback(channel):
    logging.info("Start button pressed")
    asyncio.create_task(handle_start_button())  # Handle start button logic asynchronously

def shutdown_button_callback(channel):
    logging.info("Shutdown button pressed")
    asyncio.create_task(handle_shutdown_button())  # Handle shutdown button logic asynchronously

def reset_button_callback(channel):
    logging.info("Reset button pressed")
    asyncio.create_task(reset_system())  # Handle reset logic asynchronously

# Attach interrupts to GPIO pins
GPIO.add_event_detect(START_PIN, GPIO.RISING, callback=start_button_callback, bouncetime=200)
GPIO.add_event_detect(SHUT_DOWN_PIN, GPIO.RISING, callback=shutdown_button_callback, bouncetime=200)
GPIO.add_event_detect(RESET_PIN, GPIO.RISING, callback=reset_button_callback, bouncetime=200)

# Button Logic
async def handle_start_button():
    """Logic for the Start button."""
    logging.info("Handling start button logic")
    pc_url = "http://192.168.137.214:5000"
    image_path = await request_image_from_pc(pc_url)
    if image_path:
        logging.info(f"Image captured: {image_path}")
        result = await send_to_cloud(image_path, 'http://192.168.137.214:5000')
        if result and result.get('result') == 'granted':
            await process_access()
        else:
            logging.warning("Access denied or no response from cloud.")

async def handle_shutdown_button():
    """Logic for the Shutdown button."""
    logging.info("Shutting down system...")
    GPIO.cleanup()
    subprocess.run(['sudo', 'poweroff'])

# Reset System Logic
async def reset_system():
    """Clears cache and reboots the system."""
    logging.info("Resetting system...")
    GPIO.cleanup()
    subprocess.run(['sudo', 'reboot'])

# Placeholder for image handling and cloud communication
async def request_image_from_pc(pc_url):
    logging.info(f"Requesting image from PC: {pc_url}")
    # Simulate image capture
    await asyncio.sleep(1)
    return "./dummy_image.jpg"

async def send_to_cloud(image_path, cloud_url):
    logging.info(f"Sending image to cloud: {image_path} -> {cloud_url}")
    # Simulate cloud response
    await asyncio.sleep(1)
    return {"result": "granted"}

async def process_access():
    """Simulates granting access with a buzzer sound."""
    logging.info("Access granted. Activating buzzer...")
    GPIO.output(BEEP_PIN, GPIO.HIGH)
    await asyncio.sleep(0.5)
    GPIO.output(BEEP_PIN, GPIO.LOW)

# Main Event Loop
async def main():
    logging.info("System is ready and waiting for button events.")
    try:
        while True:
            await asyncio.sleep(1)  # Keep the loop alive
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
