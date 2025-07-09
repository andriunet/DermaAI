from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from predictor import predict_skin_condition_local

app = FastAPI()

# Permitir CORS (opcional si lo vas a consumir desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = predict_skin_condition_local(file_path)

    response = []
    for name, prob, description in results:
        response.append({
            "condition": name,
            "probability": prob,
            "description": description
        })

    return {"results": response}