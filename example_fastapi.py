from huggingface_hub import login
login(token="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")

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

import io
import cv2
import numpy as np
import base64

app = FastAPI()

model_id = "google/paligemma-3b-mix-224"
device = "cuda"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

@app.post("/predict")
async def predict_image(image_data: dict):
    try:
        print(image_data['image'][:80])
        # Convert base64 image data to OpenCV image
        image_data = base64.b64decode(image_data['image'].split(",")[1])
        
        # Convert the binary data to a NumPy array
        image_array = np.frombuffer(image_data, np.uint8)

        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert the OpenCV image to PIL image format
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        prompt = "caption en"
        model_inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            print(decoded)
        
        result_text = decoded
        return {"text": result_text}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error processing image: " + str(e))
        
# Mount the 'static' directory to serve HTML files
app.mount("/", StaticFiles(directory="static"), name="static")