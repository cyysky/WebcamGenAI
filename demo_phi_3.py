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
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import base64
import cv2
import numpy as np

app = FastAPI()

model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

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
        
        message = [ 
                    {"role": "user", "content": f"<|image_1|>\n{image_data['prompt']}"}
                ] 
        
        prompt = processor.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, [pil_image], return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        print(response)
        result_text = response
        
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

