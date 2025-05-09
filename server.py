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

from src.image_processor import ImageProcessor
from src.constants import BASE_MODEL_PATH, LORA_PATH, BASE_IMAGE_URL

RETRY_MS = int(os.environ.get("RETRY_MS", 500))
RETRY_MAX = int(os.environ.get("RETRY_MS", 1200))

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
_base_image = Image.open(BytesIO(urlopen(BASE_IMAGE_URL).read()))

def initialize():
    global _lock, _processor, _is_ready
    _processor = ImageProcessor(BASE_MODEL_PATH, LORA_PATH, _base_image)
    with _lock:
        _is_ready = True

def process_prompt(url, workflow_id):
    global _lock, _processor, _is_ready
    retries = 0
    while retries < RETRY_MAX:
        retries += 1
        with _lock:
            if _is_ready:
                break
        
        time.sleep(RETRY_MS / 1000.)
    else:
        raise RuntimeError("initilize time out")

    result_image = _processor.process_image(_lora_names[workflow_id], subject_imgs=[Image.open(BytesIO(urlopen(url).read()))])

    jpg_buffer = BytesIO()
    result_image.save(jpg_buffer, format='JPEG', quality=85)
    
    return jpg_buffer

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_thread = Thread(target=initialize)
    init_thread.daemon = True
    init_thread.start()

    yield

    pass
        
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/process')
async def process(query: dict):
    try:
        url = query.get("url", ORIGIN_IMAGE_URL)
        workflow_id = query.get("workflow_id", 1)

        output = process_prompt(url, workflow_id)

        return Response(
            content=output,
            media_type=f"image/jpeg"
        )

    except Exception as e:  
        raise HTTPException(
            status_code=500,
            detail=f"Error during job execution: {str(e)}"
        )

if __name__ == "__main__":
    init_thread = Thread(target=initialize)
    init_thread.daemon = True
    init_thread.start()

    import uvicorn
    uvicorn.run(
        app="server:app",
        host="0.0.0.0",
        port=8188,
        reload=False
    )