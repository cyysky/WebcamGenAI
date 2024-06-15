import ssl
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('cert.pem', keyfile='key.pem')

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse

import os
from PIL import Image 
import requests 
from transformers import AutoModel, AutoTokenizer
import base64
import cv2
import numpy as np
import torch

app = FastAPI()

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

@app.post("/predict")
async def predict_image(image_data: dict):
    try:
        print(image_data['image'][:80])
        # Convert base64 image data to OpenCV image
        image_decoded = base64.b64decode(image_data['image'].split(",")[1])
        
        # Convert the binary data to a NumPy array
        image_array = np.frombuffer(image_decoded, np.uint8)

        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert the OpenCV image to PIL image format
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
        msgs = [{'role': 'user', 'content': image_data['prompt']}]

        res = model.chat(
            image=pil_image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
        print(res)
        result_text = res
        return {"text": result_text}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error processing image: " + str(e))
        
# Mount the 'static' directory to serve HTML files
#app.mount("/", StaticFiles(directory="static",html = True), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join("static", "demo.html")) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

