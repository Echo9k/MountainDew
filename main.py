import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# OpenVINO Inference Engine imports (install openvino and openvino-dev)
from openvino.runtime import Core

app = FastAPI()

# Load your OpenVINO IR model
# NOTE: Update the paths to your actual model.xml/model.bin files
ie_core = Core()
model = ie_core.read_model(model="model/model.xml", weights="model/model.bin")
compiled_model = ie_core.compile_model(model=model, device_name="CPU")

# If you have specific input and output layer names, set them here
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

class InferenceRequest(BaseModel):
    b64_image: str = None

@app.get("/")
def root():
    return {"message": "Vision-Language Model API is up and running!"}

@app.post("/infer")
def infer_image(
    file: UploadFile = File(default=None),
    request_body: InferenceRequest = Body(default=None)
):
    """
    Inference endpoint that accepts either:
    1. An uploaded image file, OR
    2. A base64-encoded image in the request body.
    """

    # 1. Check if a file was uploaded
    if file is not None:
        try:
            contents = file.file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading image file: {str(e)}")

    # 2. Otherwise, check for base64 input in JSON
    elif request_body and request_body.b64_image is not None:
        try:
            decoded_bytes = base64.b64decode(request_body.b64_image)
            image = Image.open(io.BytesIO(decoded_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding base64 string: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Please upload a file or send a base64 string."
        )

    # Preprocessing: adapt as needed for your model
    # Example: convert to RGB, resize, etc.
    image = image.convert("RGB")
    # Suppose the model expects 224x224 input
    image = image.resize((224, 224))

    # Convert PIL image to numpy array and transpose if needed
    import numpy as np
    image_array = np.array(image, dtype=np.float32)
    # Example normalization or scaling if required by your model:
    # image_array = image_array / 255.0
    # Transpose to [N, C, H, W] if your model expects channels-first
    image_array = np.transpose(image_array, (2, 0, 1))
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Perform inference
    # For a typical vision model, you'd get something like classification or bounding boxes back
    # For a vision-language model, you might get text embeddings or direct text output
    results = compiled_model([image_array])[output_layer]
    
    # Here you would parse `results` to generate your textual output.
    # This is highly model-dependent. We'll mock it for illustration.
    # Suppose your model returns text tokens in `results`.
    # In a real scenario, you'd decode or postprocess these tokens:
    fake_caption = "This is a placeholder text output from the vision-language model."

    return JSONResponse(content={"inference_result": fake_caption})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
