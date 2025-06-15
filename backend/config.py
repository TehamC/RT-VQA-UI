import os

# Get the directory where config.py is located
# This makes the paths relative to the project, which is more robust
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(CURRENT_DIR, "uploads")
YOLO_MODEL_PATH = os.path.join(CURRENT_DIR, "trained_models", "best.pt")
TRAINED_MODEL_PATH = os.path.join(CURRENT_DIR, "trained_models", "Q6_context")
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 

