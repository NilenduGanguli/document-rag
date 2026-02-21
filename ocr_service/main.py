from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, AutoModelForCausalLM
import transformers.dynamic_module_utils as dmu
from PIL import Image
import io
import torch
import os

app = FastAPI(title="LLM OCR Service")

model_id = "microsoft/Florence-2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Bypass flash_attn requirement for Florence-2 on CPU/environments without flash_attn
dmu.check_imports = lambda filename: []

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    prompt = "<OCR>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    
    return {"text": parsed_answer.get("<OCR>", "")}
