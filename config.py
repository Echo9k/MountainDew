# config.py

# Model configuration
MODEL_ID = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"

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
DEFAULT_PROMPT = "<|image_1|>\nWhat is unusual on this picture?"
