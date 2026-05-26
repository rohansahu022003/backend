from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Hugging Face Model Repository
model_name = "rohansahu02/bart-large-summarizer"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

# Request body
class SummaryRequest(BaseModel):
    text: str

# Summarization endpoint
@app.post("/summarize")
def summarize(req: SummaryRequest):

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return {
        "summary": summary
    }