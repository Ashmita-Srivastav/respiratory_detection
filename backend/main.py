from fastapi import FastAPI, UploadFile
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