import numpy as np, uvicorn, io, cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from UNet_Model import *

UNET = u_net(2, (256, 256, 3), 45, base_filters = 16)

UNET.load_weights("model/car_segmentation_model")

app = FastAPI(title='Image Segmentation Model')

@app.get("/")
async def home():
    return "Docs: http://localhost:8000/docs."

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

    if not fileExtension:
        raise HTTPException(status_code = 415, detail = "Unsupported file type")

    image_stream = io.BytesIO(file.file.read())

    image_stream.seek(0)

    # Use pillow image for consistency
    # (cv2 internally uses BGR ordering while pillow uses RGB, just stay consistent with color order)
    image_arr = np.array(Image.open(image_stream))
    
    res = UNET(process_to_tensor(image_arr, 256, 256))

    im = np.repeat(np.round(res.numpy()) * 255., 3, axis = -1).squeeze().astype("uint8")

    cv2.imwrite(f"images_uploaded/{filename}", im)

    file_image = open(f"images_uploaded/{filename}", mode = "rb")

    return StreamingResponse(file_image, media_type = "image/jpeg")

if __name__ == "__main__":
    host = "127.0.0.1"

    uvicorn.run(app, host = host, port = 8000)

