# Fine-Tuning DistilBERT for Sentiment Analysis

This repository contains the code and workflow for fine-tuning the DistilBERT (distilbert-base-uncased) model from Hugging Face on a sentiment analysis task. The dataset used for training is sourced from Kaggle.

## 🚀 Features
Fine-tuning distilbert-base-uncased for sentiment classification.
Data preprocessing and tokenization using Hugging Face Transformers.
Model training and evaluation.
Inference script to predict sentiment on new text samples.

## 📂 Dataset
The 3 datasets used for fine-tuning is available on Kaggle. You can download it using below links:
    1. IMDB dataset (Sentiment analysis) in CSV format [link](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format?select=Test.csv)
    2. Sentiment Analysis Dataset [link](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv)
    3. Stock News Sentiment Analysis(Massive Dataset) [link](https://www.kaggle.com/datasets/avisheksood/stock-news-sentiment-analysismassive-dataset)
    4.final dataset in Huggingface [link](https://huggingface.co/datasets/KaushiGihan/sentiment_analys_3_combine_ds)


## 📊 Data Preprocessing & Visualization
The dataset is cleaned, preprocessed, and visualized using Pandas, Matplotlib, and Seaborn. Open and run the notebook:
 

📜 Notebook: notebooks/data_preprocessing.ipynb

## 📦 Installation
Clone the repository and install the required dependencies:

    ``` python 
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    pip install -r requirements.txt
    ```
## 🛠 Training the Model
The DistilBERT model is fine-tuned using Hugging Face's Transformers library. Training includes early stopping, learning rate scheduling, and evaluation metrics. Open and run the notebook:

1. 📜 Notebook: notebooks/model_training.ipynb

2.Alternatively, run the training script:

    ``` python 
    python train.py
    ```
## 🔍 Inference
To test the model on new text inputs, run:

    ``` python 
    python app.py 
    ```

    
    ![image](https://github.com/KaushiML3/Fine-tuning-a-LLM-for-sentiment-analysis/blob/main/img/Screenshot%20(104).png)
    ![image](https://github.com/KaushiML3/Fine-tuning-a-LLM-for-sentiment-analysis/blob/main/img/Screenshot%20(105).png)

## 📄 Acknowledgments

1. Hugging Face for the DistilBERT model.
2. Kaggle for the dataset.
