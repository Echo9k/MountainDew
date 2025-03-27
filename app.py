import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Import snapshot_download to pin model version and set a cache directory.
from huggingface_hub import snapshot_download

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer

# Import external configuration.
from config import (
    MODEL_ID,
    MODEL_REVISION,
    CACHE_DIR,
    GENERATION_ARGS,
    SERVER_HOST,
    SERVER_PORT,
    DEFAULT_PROMPT,
    MAX_IMAGE_SIZE,
    NORMALIZE_IMAGE,
)

# Download the model snapshot once (if not already in CACHE_DIR)
model_dir = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_DIR)

app = FastAPI()

# For JSON-based requests.
class InferenceRequest(BaseModel):
    b64_image: str = None
    prompt: str = None

# Load the processor and model from the downloaded directory.
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir, trust_remote_code=True)

def optimized_preprocess_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE, normalize: bool = NORMALIZE_IMAGE) -> Image.Image:
    """
    Optimize image preprocessing by:
      - Converting to RGB.
      - Resizing if the largest dimension exceeds max_size.
      - Optionally normalizing pixel values to [0, 1].
    """
    image = image.convert("RGB")
    
    # Resize image if necessary.
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Optionally normalize: if your model's processor handles normalization, you may skip this.
    if normalize:
        image_array = np.array(image, dtype=np.float32) / 255.0
        # Convert back to PIL image if necessary.
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

    # Optimize image preprocessing.
    image = optimized_preprocess_image(image)

    if prompt is None:
        prompt = DEFAULT_PROMPT

    # Preprocess inputs for the vision-language model.
    inputs = ov_model.preprocess_inputs(text=prompt, image=image, processor=processor)

    # Set generation arguments (adding streamer).
    generation_args = {
        **GENERATION_ARGS,
        "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    }

    # Generate model output.
    generate_ids = ov_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return JSONResponse(content={"inference_result": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
