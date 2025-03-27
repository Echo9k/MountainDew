import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer

app = FastAPI()

# For JSON-based requests.
class InferenceRequest(BaseModel):
    b64_image: str = None
    prompt: str = None

model_id = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
ov_model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)

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

    # Use a default prompt if none is provided.
    if prompt is None:
        prompt = "<|image_1|>\nWhat is unusual on this picture?"

    # Preprocess inputs for the model.
    inputs = ov_model.preprocess_inputs(text=prompt, image=image, processor=processor)
    generation_args = { 
        "max_new_tokens": 50, 
        "temperature": 0.0, 
        "do_sample": False,
        "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    }

    # Generate response.
    generate_ids = ov_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return JSONResponse(content={"inference_result": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
