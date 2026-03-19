from fastapi import FastAPI, UploadFile
import cv2
import numpy as np

from inference.pipeline import run_pipeline
from explainability.gradcam import generate_gradcam
from llm.ollama_service import ask_llm

app = FastAPI(title="Glaucoma AI System")


@app.post("/predict")
async def predict(file: UploadFile):

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    result = run_pipeline(image)

    heatmap = generate_gradcam(image)

    return {
        "prediction": result["prediction"],
        "cdr": result["cdr"],
        "vessel_risk": result["vessel_risk"],
        "gradcam": heatmap
    }


@app.post("/ask")
def ask(question: str):
    response = ask_llm(question)
    return {"answer": response}