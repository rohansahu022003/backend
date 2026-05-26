from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

headers = {
    "Authorization": "Bearer " + HF_TOKEN
}

class SummaryRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/summarize")
def summarize(req: SummaryRequest):

    payload = {
        "inputs": req.text
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    result = response.json()

    print(result)

    if isinstance(result, list):
        return {
            "summary": result[0]["summary_text"]
        }

    return {
        "error": result
    }