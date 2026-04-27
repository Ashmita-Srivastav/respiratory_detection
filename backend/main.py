from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile):
    y, sr = librosa.load(file.file)
    
    # your model logic
    prediction = "Healthy"

    return {"prediction": prediction}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
