Raspberry Pi Face Recognition System:

This repository contains the implementation of a camera-based face recognition system using Raspberry Pi. The system captures images using the Raspberry Pi Camera Module, processes them to detect faces, and identifies individuals using advanced machine learning models. It integrates seamlessly with a cloud-based server for enhanced processing capabilities.

Features:

        * Real-Time Face Recognition: Detects and recognizes faces in real-time using Raspberry Pi.
        * Cloud Integration: Offloads heavy computation tasks to a cloud server.
        * Local Caching: Utilizes FAISS for fast local recognition of known faces.
        * Hardware Integration: Includes GPIO-based controls for buttons and buzzers.
        * Extensibility: Modular design allows integration with different face detection models.

Requirements:

        Hardware:

                Raspberry Pi 4 (or newer) with Raspbian OS
                Raspberry Pi Camera Module (v2 recommended)
                Buttons and a buzzer connected to GPIO pins
        Software:
        
                Python 3.9 or newer
                Required libraries:
                RPi.GPIO for hardware control
                httpx for HTTP communication
                OpenCV for image processing
                FAISS for fast similarity search
                Dlib or YOLO for face detection and embedding extraction
                asyncio for asynchronous operations
                Cloud server with an API for face recognition (e.g., Flask, FastAPI, or Quart)

Usage:

        1. Start the Raspberry Pi Face Recognition Script
        Run the Python script on your Raspberry Pi:
                python3 code_on_hardware.py
                
        2. API Endpoint for Cloud Integration
        Ensure the cloud server is running. Update the pc_url and cloud_ai_url in the code with the server's IP and port.
        Run the code that is for cloud:
                python3 main.py
