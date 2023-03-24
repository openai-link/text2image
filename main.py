import time
import random
import torch
import uvicorn
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# http service fastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/imgs", StaticFiles(directory="imgs"), name="imgs")
app.mount("/views", StaticFiles(directory="views"), name="views")

# load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


@app.post("/text2image")
async def text2image(text: str = Form(...)):
    image = pipe(text).images[0]
    filename = f"{int(time.time())}{random.randint(1000, 9999)}.png"
    path = f"imgs/{filename}"
    image.save(path)
    return {"image": path, "text": text}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                log_level="info", reload=True)
