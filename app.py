import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer

# Import external configuration.
from config import (
    MODEL_ID,
    GENERATION_ARGS,
    SERVER_HOST,
    SERVER_PORT,
    DEFAULT_PROMPT,
    MAX_IMAGE_SIZE,
    NORMALIZE_IMAGE,
)

app = FastAPI()

# For JSON-based requests.
class InferenceRequest(BaseModel):
    b64_image: str = None
    prompt: str = None

# Initialize the processor and the model using the configuration.
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
ov_model = OVModelForVisualCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

def optimized_preprocess_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE, normalize: bool = NORMALIZE_IMAGE) -> Image.Image:
    """
    Optimize image preprocessing by:
      - Converting to RGB,
      - Resizing the image if its largest dimension exceeds max_size,
      - Optionally normalizing pixel values to [0,1].
    
    Returns a PIL image (normalization is applied to a numpy array if needed).
    """
    # Convert image to RGB.
    image = image.convert("RGB")
    
    # Resize image if larger than max_size.
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Optional normalization: convert to numpy array and scale pixel values.
    if normalize:
        image_array = np.array(image, dtype=np.float32) / 255.0
        # If your model's preprocessor expects a PIL image, you might want to convert back.
        # Here, we'll assume the processor accepts normalized arrays.
        # Otherwise, comment out the next line if the processor does its own normalization.
        image = Image.fromarray((image_array * 255).astype(np.uint8))
    
    return image

@app.get("/")
def root():
    return {"message": "Vision-Language Model API is up and running!"}

@app.post("/infer")
async def infer_image(request: Request):
    content_type = request.headers.get("Content-Type", "")
    image = None
    prompt = None

    if "multipart/form-data" in content_type:
        # Handle multipart/form-data: file upload or base64 string provided as form fields.
        form = await request.form()
        file = form.get("file")
        prompt = form.get("prompt")
        b64_image = form.get("b64_image")

        if file is not None and hasattr(file, "file"):
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading image file: {str(e)}")
        elif b64_image is not None:
            try:
                decoded_bytes = base64.b64decode(b64_image)
                image = Image.open(io.BytesIO(decoded_bytes))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error decoding base64 string from form: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No image provided in form data.")
            
    elif "application/json" in content_type:
        # Handle JSON payload.
        try:
            json_body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        b64_image = json_body.get("b64_image")
        prompt = json_body.get("prompt")
        if b64_image is not None:
            try:
                decoded_bytes = base64.b64decode(b64_image)
                image = Image.open(io.BytesIO(decoded_bytes))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error decoding base64 string from JSON: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No image provided in JSON.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported Content-Type.")

    if image is None:
        raise HTTPException(status_code=400, detail="No image provided.")

    # Optimize the image before processing.
    image = optimized_preprocess_image(image)

    # Use the provided prompt or fall back to the default prompt from the config.
    if prompt is None:
        prompt = DEFAULT_PROMPT

    # Preprocess inputs for the vision-language model.
    inputs = ov_model.preprocess_inputs(text=prompt, image=image, processor=processor)

    # Set generation arguments from config, and add streamer.
    generation_args = {
        **GENERATION_ARGS,
        "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    }

    # Generate response.
    generate_ids = ov_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return JSONResponse(content={"inference_result": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
