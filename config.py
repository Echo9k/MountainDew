# config.py

# Model configuration
MODEL_ID = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
MODEL_REVISION = "main"  # Replace with a specific commit hash or tag to pin the version
CACHE_DIR = "./model"

# Generation settings
GENERATION_ARGS = {
    "max_new_tokens": 50,
    "temperature": 0.0,
    "do_sample": False,
}

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Default prompt if none is provided
DEFAULT_PROMPT = "<|image_1|>\nWhat is the main shape in this picture? [single world if possible]"

# Preprocessing configuration
MAX_IMAGE_SIZE = 420  # Maximum dimension (width or height) for the image
NORMALIZE_IMAGE = True  # Whether to normalize image pixel values to [0, 1]
