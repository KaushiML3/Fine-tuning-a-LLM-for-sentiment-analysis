from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse , JSONResponse,HTMLResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import uvicorn
import os
import torch
import torchvision

from src.custom_logger import setup_logger
logger=setup_logger("app")


app=FastAPI(title="Sentiment",
    description="FastAPI",
    version="0.115.4")

# Allow all origins (replace * with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get("/")
async def root():
  return {"Fast API":"API is woorking"}

id2label={0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

try:
    model_id="KaushiGihan/distilbert-base-uncased-sentiment_analysis_model"
    config = PeftConfig.from_pretrained(model_id)
    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(inference_model, model_id)

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")


@app.post("/sentiment_anlys")    
async def sentiment_anlys(input_text:str):
    try:

        inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512     
        ).to("cpu")  

        logits = model(inputs).logits
        predictions = torch.max(logits,1).indices
    
        #print(input_text+ " - " + id2label[predictions.tolist()[0]])

        return {"statue":1,"input_text":input_text,"Message":id2label[predictions.tolist()[0]]}

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"statue":0,"input_text":input_text,"Message":str(e)}

if __name__=="__main__":
    
    uvicorn.run("app:app",host="0.0.0.0", port=8000, reload=True)