# SENTIMENT-ANALYSIS

This project compares sentiment analysis performance across three NLP models: VADER, RoBERTa, and a Logistic Regression TF-IDF model that I built. I evaluate each model using precision, recall, F1 score, and accuracy to understand how well they classify sentiment.

## Libraries and Tools

### VADER (Rule-Based Model)
```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
```

### RoBERTa (Pretrained Transformer Model)
```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
```

### Logistic Regression with TF-IDF (Machine Learning Model)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
```

### Graphing and Data Cleaning
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Findings

When I compared the models, I noticed that the machine learning approaches generally did better as the dataset got larger. The Logistic Regression model with TF-IDF actually performed the best overall, beating both VADER and RoBERTa in accuracy, precision, recall, and F1 score.

This shows that a straightforward ML model can still do really well, and in some cases even better than transformer models, especially when TF-IDF is a good fit for the type of text being analyzed.
