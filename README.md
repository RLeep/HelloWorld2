# Rachel Lee: Division Sigma
Sentiment analysis was performed using Logistic Regression. Various machine learning algorithms were used for sentiment analysis on social media status updates and compared based on the evaluation metric. Initially, I tried deep learning techniques such as XGBoost and LSTM. Deep learning is one of the most advanced and recent ML methods that are powerful because of their hidden layers. However, both did not significantly improve accuracy while being computationally expensive to train. I finally tried Logistic Regression which improved accuracy and was able to train in a reasonable amount of time. Logistic regression is a simple yet efficient machine learning algorithm for sentiment analysis. 

## Motivation
Sentiment analysis is a NLP task that aims to classify a piece if text as either positive or negative in sentiment. Sentiment analysis allows business to make sense of unstructured text and understand the social sentiment of their brand in order to improve. However, sentiment analysis faces challenges due to the use of sarcasm in everyday speech, word ambiguity and grammatical and spelling mistakes.

## Instillation
As this is a more involved python program, a virtual environment was used. A pip freeze was used to save the installed packages. To install the necessary libraries, use 
```bash
pip install -r requirements.txt
```
Or first ensure pip is up to date (`pip install --upgrade pip`) and install wheel (`pip install wheel`) before installing nltk, numpy, scikit-learn, and spacy (`pip install nltk numpy tensorflow scikit-learnspacy`). For spacy, you will need to get the spacy English model (`python -m spacy download en_core_web_sm`)

## Data Cleansing
Preprocessing is a vital part of machine learning to get clean data. Data cleansing improves data quality and improves productivity by leaving only significant and meaningful data. In this implementation for statistical analysis, phrases such as tags and hyperlinks were removed from the text as they rarely signified sentiment. Emoticons and other unusual characters were removed for data clarity. 

## Authors
* Rachel Lee
