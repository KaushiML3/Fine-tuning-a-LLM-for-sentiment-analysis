from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification,DataCollatorWithPadding,TrainingArguments,Trainer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import evaluate
import torch
import torchvision
import numpy as np
import os

from src.custom_logger import setup_logger
from src.preprocess import datapreprocess
from src.utility import push_hub
#from src import setup_logger

logger=setup_logger("train")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)  # Convert logits to label predictions

    acc = (predictions == labels).mean()
    return {"eval_accuracy": float(acc)}


def train_model(model,tokenizer,tokenized_dataset,base_model,model_save_name):
  try:

    peft_config = LoraConfig(task_type="SEQ_CLS",
                            r=4,
                            lora_alpha=32,
                            lora_dropout=0.01,
                            target_modules = ['q_lin'])


    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ud=os.path.join("artifact", base_model + "-" + model_save_name)
    # define training arguments
    training_args = TrainingArguments(
        output_dir = ud,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
    )



    # creater trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
        compute_metrics=compute_metrics,
    )
    # train model
    trainer.train()

    push_hub(model,trainer,base_model,model_save_name)

  except Exception as e:
    logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":

  lr = 1e-3
  batch_size = 32
  num_epochs = 10
  label=['Negative', 'Positive']
  dataset_path='KaushiGihan/sentiment_analys_3_combine_ds'#'shawhin/imdb-truncated'
  base_model='distilbert-base-uncased'
  model_save_name="sentiment_analysis_model"


  try:
    #data preprocessing
    logger.info(f"Data preprocessing start")
    pre=datapreprocess(dataset_path)
    model,tokenizer,tokenized_dataset=pre.preprocess_function(base_model,label)
    logger.info(f"Data preprocessing Done")

    #model Training
    logger.info(f"Model training start")
    train_model(model,tokenizer,tokenized_dataset,base_model,model_save_name)
    logger.info(f"Model training Done")

  except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
