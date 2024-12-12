import cv2
import torch
from torchvision import transforms
import numpy as np
import pathlib,platform

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Define global variables for model and device
global_model = None
global_device = None

def get_device():
    """
    Determines the device (CPU or GPU) to be used for computation.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_once():
    """
    Loads the YOLO model once and stores it globally for reuse.

    Returns:
        None
    """
    global global_model, global_device
    if global_model is None:
        try:
            global_device = get_device()
            model_path = r"cloud/ml_yolov5/best.pt"
            global_model = torch.hub.load(
                r'cloud/ml_yolov5/content/yolov5',
                'custom',
                path=model_path,
                device=global_device,
                source='local',
                force_reload=True
            )
            print(f"YOLO model loaded on {global_device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

def detect_faces(image_path):
    """
    Performs face detection on an input image using the YOLO model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        results: The detection results containing bounding boxes and class scores.
    """
    global global_model, global_device
    if global_model is None:
        raise RuntimeError("Model is not loaded. Call load_model_once() first.")

    # Ensure the model is on the correct device
    global_model.to(global_device)

    # Read the input image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at path: {image_path}")

    # Perform inference on the image
    try:
        results = global_model(image)
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")

    return results

def call_ML(image_path):
    """
    Main function to perform face detection using the preloaded model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        results: The detection results from the YOLO model.
    """
    if not image_path:
        raise ValueError("Image path is required.")
    return detect_faces(image_path)

def extract_face_embeddings(face, embedding_model, device):
    """
    Extracts face embeddings from a cropped face image using a pre-trained embedding model.

    Args:
        face (np.ndarray): The cropped face image.
        embedding_model: The pre-trained model to extract embeddings.
        device (torch.device): The device to run the embedding model on.

    Returns:
        np.ndarray: The extracted face embedding as a NumPy array.
    """
    if face is None or not isinstance(face, np.ndarray):
        raise ValueError("Invalid face input: Expected a numpy array.")
    if embedding_model is None:
        raise ValueError("Embedding model is not provided.")
    
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Resize((160, 160)),          # Resize to the input size expected by the embedding model
        transforms.Normalize([0.5], [0.5])      # Normalize pixel values to a fixed range
    ])
    
    # Apply preprocessing to the face image and add a batch dimension
    face_tensor = preprocess(face).unsqueeze(0).to(device)
    
    # Perform inference without tracking gradients
    with torch.no_grad():
        try:
            embedding = embedding_model(face_tensor)  # Get the face embedding
        except Exception as e:
            raise RuntimeError(f"Error during embedding extraction: {e}")
    
    return embedding.cpu().numpy()  # Convert the embedding tensor to a NumPy array and return
