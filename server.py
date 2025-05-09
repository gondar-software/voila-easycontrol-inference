import runpod
import os
import time
import base64
from threading import Thread, Lock
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from src.image_processor import ImageProcessor
from src.constants import BASE_MODEL_PATH, LORA_PATH, BASE_IMAGE_URL

RETRY_MS = int(os.environ.get("RETRY_MS", 500))
RETRY_MAX = int(os.environ.get("RETRY_MAX", 500))

_lora_names = {
    1: "Ghibli.safetensors",
    2: "snoopy-1500.safetensors",
    4: "3d-cartoon-960.safetensors",
    5: "labubu-660.safetensors",
    6: "classic_toys.safetensors",
}
_lock = Lock()
_processor: Optional[ImageProcessor] = None
_is_ready = False

try:
    _base_image = Image.open(BytesIO(urlopen(BASE_IMAGE_URL).read()))
except Exception as e:
    raise RuntimeError(f"Failed to load base image from {BASE_IMAGE_URL}: {str(e)}")

def initialize():
    global _processor, _is_ready, _lock
    try:
        _processor = ImageProcessor(BASE_MODEL_PATH, LORA_PATH, _base_image)
        with _lock:
            _is_ready = True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ImageProcessor: {str(e)}")

def process_prompt(url: str, workflow_id: int) -> BytesIO:
    global _processor, _is_ready, _lock
    
    retries = 0
    while retries < RETRY_MAX:
        with _lock:
            if _is_ready and _processor is not None:
                break
        time.sleep(RETRY_MS / 1000.)
        retries += 1
    else:
        raise RuntimeError(f"Initialization timeout after {RETRY_MAX * RETRY_MS / 1000} seconds")

    try:
        input_image = Image.open(BytesIO(urlopen(url).read()))
        if workflow_id not in _lora_names:
            raise ValueError(f"Invalid workflow_id: {workflow_id}")
        
        lora_name = _lora_names[workflow_id]
        result_image = _processor.process_image(
            lora_name, 
            subject_imgs=[input_image]
        )

        jpg_buffer = BytesIO()
        result_image.save(jpg_buffer, format='JPEG', quality=85)
        jpg_buffer.seek(0)
        
        return jpg_buffer
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting initialization thread...")
    init_thread = Thread(target=initialize)
    init_thread.daemon = True
    init_thread.start()

    yield

    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health_check():
    return {"status": "ready" if _is_ready else "initializing"}

@app.post('/process')
async def process(query: dict):
    try:
        url = query.get("url")
        workflow_id = query.get("workflow_id", 1)
        
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        if workflow_id not in _lora_names:
            raise HTTPException(status_code=400, detail="Invalid workflow_id")

        output = process_prompt(url, workflow_id)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during processing: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="server:app",
        host="0.0.0.0",
        port=8188,
        reload=False
    )