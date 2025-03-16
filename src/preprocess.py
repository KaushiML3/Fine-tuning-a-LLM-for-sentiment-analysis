from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from src.custom_logger import setup_logger
#from src import setup_logger
logger=setup_logger("preprocess")

class datapreprocess():
  def __init__(self,dataset_path):
    self.dataset_path=dataset_path
    self.dataset = load_dataset(self.dataset_path)
    self.model = None
    self.tokenizer = None
    self.tokenized_dataset=None

  def tokenize_function(self,examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    self.tokenizer.truncation_side = "left"
    tokenized_inputs = self.tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    return tokenized_inputs


  def preprocess_function(self,base_model,label):
    try:

      id2label = dict(zip(range(len(label)), label))
      label2id = dict(zip(label,range(len(label))))

      self.model = AutoModelForSequenceClassification.from_pretrained(base_model, 
                                                                      num_labels=2,
                                                                      id2label=id2label,
                                                                      label2id=label2id)
      
      self.tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)

      # add pad token if none exists
      if self.tokenizer.pad_token is None:
          self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
          model.resize_token_embeddings(len(self.tokenizer))

      self.tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)
      data=self.tokenized_dataset
      model=self.model
      tokenizer=self.tokenizer
    
      return self.model,self.tokenizer,self.tokenized_dataset
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None

