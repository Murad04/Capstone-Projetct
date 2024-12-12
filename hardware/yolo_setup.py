import torch
from torchvision import transforms
import cv2
from base_logger import log_function

# Path to the pre-trained face recognition model
model_path = "hardware\best.pt"  # Path to the .pt file containing the model
device = "cpu"  # Device to load the model on ('cpu' or 'cuda' for GPU)
global model, transform  # Declare global variables for the model and transformation pipeline

@log_function
async def load_model():
    """
    Loads the pre-trained face recognition model and sets up the transformation pipeline.
    """
    global model, transform
    # Load the PyTorch model from the specified path and map it to the chosen device
    model = torch.load(model_path, map_location=device,weights_only=True)
    model.eval()  # Set the model to evaluation mode (disables training-specific layers like dropout)
    
    # Define the transformation pipeline to preprocess the input image
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the image to PIL format (required for certain transformations)
        transforms.Resize((224, 224)),  # Resize the image to the expected input size for the model
        transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
    ])


@log_function
async def get_face_embedding(image_path):
    """
    Processes an image to generate its face embedding using the loaded model.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Flattened face embedding vector.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        # Raise an error if the image cannot be loaded
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Apply the transformation pipeline and add a batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)  
    
    # Disable gradient computation during inference for efficiency
    with torch.no_grad():
        # Pass the image tensor through the model to get the embedding
        embedding = model(img_tensor).cpu().numpy().flatten()  # Convert the tensor to a NumPy array and flatten it
    
    return embedding  # Return the embedding vector

if __name__ == "__main__":
    import asyncio  # Import asyncio for running the asynchronous function
    
    @log_function
    async def main():
        """
        Main function to load the model, process an image, and generate its face embedding.
        """
        load_model()  # Load the face recognition model and set up the transformation pipeline
        image_path = "path/to/face_image.jpg"  # Specify the path to the input image
        embedding = await get_face_embedding(image_path)  # Generate the face embedding
        print("Face Embedding:", embedding)  # Print the resulting face embedding vector
    
    asyncio.run(main())  # Run the asynchronous main function