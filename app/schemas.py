from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):

    prediction: str
    cdr: float
    vessel_risk: float


class QuestionRequest(BaseModel):

    question: str


class LLMResponse(BaseModel):

    answer: str


class PatientRecord(BaseModel):

    patient_id: str
    cdr: float