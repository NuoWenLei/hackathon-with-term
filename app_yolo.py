import numpy as np, uvicorn, io, cv2, pyngrok, nest_asyncio
import tensorflow
import requests
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = "FastAPI Yolo")

origins = [
    "http://localhost",
    "http://localhost:8080",
	"http://localhost:60493"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return "Docs: http://localhost:8000/docs."

@app.post("/predict_url")
def url_prediction(url: str = "url"):
	print(url)
	image_stream = requests.get(fr"{url}", stream = True).raw

	file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	
	bbox, label, conf = cv.detect_common_objects(image, model="yolov3-tiny")

	return JSONResponse({
		"number_of_cars": len([l for l in label if l == "car"]),
	"estimated_gas_emission_per_year": f"{4.6 * len([l for l in label if l == 'car']):.2f} metric tons of carbon dioxide emission per year"
	})
	# According to the United States Environmental Protection Agency

# Prediction API path
@app.post("/predict") 
def prediction(write_conf: str = "False", file: UploadFile = File(...)):

	# input file stuff
    filename = file.filename
    write_conf_bool = bool(write_conf)
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")


    
	# read file as stream
    image_stream = io.BytesIO(file.file.read())
    
    image_stream.seek(0)
    
    # stream to byte array to numpy array as uint8
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # cv2 IMREAD color is BGR format
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    bbox, label, conf = cv.detect_common_objects(image, model="yolov3-tiny")
    
    output_image = draw_bbox(image, bbox, label, conf, write_conf = write_conf_bool)
    
    cv2.imwrite(f'images_uploaded/{filename}', output_image)
    
    # Open the saved image for reading in binary mode
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")

if __name__ == "__main__":
	from pyngrok import ngrok
	ngrok_tunnel = ngrok.connect(8000)
	print('Public URL:', ngrok_tunnel.public_url)
	nest_asyncio.apply()
	uvicorn.run(app, port=8000)
    # host = "127.0.0.1"

    # uvicorn.run(app, host = host, port = 8000)

