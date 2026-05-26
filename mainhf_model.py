from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/t5-small"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class SummaryRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: SummaryRequest):

    payload = {
        "inputs": "summarize: " + req.text
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    result = response.json()

    try:
        summary = result[0]["summary_text"]
    except:
        summary = str(result)

    return {
        "summary": summary
    }