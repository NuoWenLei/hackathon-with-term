import numpy as np, cv2
import requests
import cvlib as cv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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