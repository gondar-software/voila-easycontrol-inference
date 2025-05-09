import runpod
import os
import time
import base64
from threading import Thread, Lock
from typing import Optional
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

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
    global _lock, _is_ready, _processor
    _processor = ImageProcessor(BASE_MODEL_PATH, LORA_PATH, _base_image)
    with _lock:
        _is_ready = True

def handler(job):
    global _lock, _is_ready, _processor

    url = job["input"].get("url", None)
    workflow_id = job["input"].get("workflow_id", 1)
    if url is None:
        return { "error": "url should be defined." }
    if _lora_names.get(workflow_id, None) is None:
        return { "error": "can't find workflow." }

    retries = 0
    while retries < RETRY_MAX:
        retries += 1
        with _lock:
            if _is_ready:
                break
        
        time.sleep(RETRY_MS / 1000.)
    else:
        return { 
            "error": "initialization time out" 
        }

    try:
        result_image = _processor.process_image(_lora_names[workflow_id], spatial_imgs=[Image.open(BytesIO(urlopen(url).read()))])

        jpg_buffer = BytesIO()
        result_image.save(jpg_buffer, format='JPEG', quality=85)
        result_base64 = base64.b64encode(jpg_buffer.getvalue()).decode('utf-8')

        return {
            "output": "success",
            "message": result_base64,
        }
    except Exception as e:
        return {
            "error": f"unknown error occurred during the processing. {e}"
        }

# Start the handler only if this script is run directly
if __name__ == "__main__":
    # Initilize processor
    init_thread = Thread(target=initialize)
    init_thread.daemon = True
    init_thread.start()

    # Handle requests
    runpod.serverless.start({"handler": handler})