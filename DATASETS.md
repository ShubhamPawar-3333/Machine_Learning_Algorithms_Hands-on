# Real-World Datasets for Machine Learning Practice

This document provides a curated list of real-world datasets for hands-on ML practice, organized by domain and difficulty level.

---

## 📥 Where to Find Datasets

### Primary Sources
- **Kaggle** - https://kaggle.com/datasets (largest collection, competitions)
- **UCI ML Repository** - https://archive.ics.uci.edu/ml (600+ datasets)
- **Google Dataset Search** - https://datasetsearch.research.google.com
- **AWS Open Data** - https://registry.opendata.aws
- **Data.gov** - Government datasets
- **Papers with Code** - https://paperswithcode.com/datasets

---

## 🏥 Healthcare & Medical

### Beginner
- **Heart Disease UCI** - Predict heart disease (303 patients, 14 attributes)
  - Source: UCI ML Repository
  - Use: Classification (Logistic Regression, Decision Trees)

- **Diabetes Dataset** - Pima Indians diabetes prediction
  - Source: Kaggle, UCI
  - Use: Binary classification

- **Breast Cancer Wisconsin** - Tumor classification (569 samples, 30 features)
  - Source: UCI, sklearn.datasets
  - Use: Binary classification

### Intermediate
- **COVID-19 Dataset** - Various datasets on Kaggle
  - Source: Kaggle
  - Use: Time series, classification

- **Medical Cost Personal Dataset** - Insurance cost prediction
  - Source: Kaggle
  - Use: Regression

---

## 💰 Finance & Business

### Beginner
- **Credit Card Fraud Detection** - Highly imbalanced (284k transactions)
  - Source: Kaggle
  - Use: Imbalanced classification, anomaly detection

- **Loan Prediction** - Dream Housing Finance dataset
  - Source: Kaggle
  - Use: Binary classification

### Intermediate
- **Stock Market Data** - Historical data from Yahoo Finance/Alpha Vantage
  - Source: Yahoo Finance API, Alpha Vantage
  - Use: Time series, regression

- **Customer Churn Prediction** - Telecom customer retention
  - Source: Kaggle
  - Use: Binary classification

- **Black Friday Sales** - Retail analytics (550k purchase records)
  - Source: Kaggle
  - Use: Regression, customer segmentation

---

## 🛒 E-commerce & Retail

### Beginner
- **Online Retail Dataset** - UCI (541k transactions)
  - Source: UCI ML Repository
  - Use: RFM analysis, customer segmentation

### Intermediate
- **Amazon Product Reviews** - Sentiment analysis (millions of reviews)
  - Source: Kaggle, AWS Open Data
  - Use: NLP, sentiment analysis

- **Instacart Market Basket Analysis** - 3M+ orders from 200k users
  - Source: Kaggle
  - Use: Recommendation systems, association rules

- **Walmart Sales Forecasting** - Time series with 45 stores
  - Source: Kaggle
  - Use: Time series forecasting

---

## 📝 Text & Social Media

### Beginner
- **Spam SMS Collection** - 5,574 SMS messages
  - Source: UCI ML Repository
  - Use: Text classification, Naive Bayes

- **IMDB Movie Reviews** - 50k reviews for sentiment analysis
  - Source: Kaggle, TensorFlow datasets
  - Use: Sentiment analysis, NLP

### Intermediate
- **Twitter Sentiment Analysis** - Real tweets with sentiment labels
  - Source: Kaggle
  - Use: NLP, sentiment classification

- **News Category Dataset** - 200k+ news headlines from HuffPost
  - Source: Kaggle
  - Use: Multi-class classification

- **Reddit Comments Dataset** - Large-scale text data
  - Source: Kaggle, pushshift.io
  - Use: NLP, topic modeling

---

## 🖼️ Computer Vision

### Beginner
- **MNIST** - 70k handwritten digits (28×28 images)
  - Source: sklearn.datasets, TensorFlow
  - Use: Image classification (beginner)

- **Fashion-MNIST** - 70k fashion product images
  - Source: Kaggle, TensorFlow
  - Use: Image classification

### Intermediate
- **CIFAR-10/100** - 60k images, 10/100 classes
  - Source: TensorFlow, PyTorch
  - Use: Image classification (more challenging)

- **Cats vs Dogs** - 25k images
  - Source: Kaggle
  - Use: Binary image classification

### Advanced
- **Chest X-Ray Images** - Pneumonia detection (5,863 images)
  - Source: Kaggle
  - Use: Medical image classification

- **Plant Disease Dataset** - 87k images of healthy/diseased plants
  - Source: Kaggle
  - Use: Multi-class image classification

---

## 📈 Time Series & Forecasting

### Beginner
- **Air Quality Dataset** - UCI (9,358 hourly observations)
  - Source: UCI ML Repository
  - Use: Time series analysis

### Intermediate
- **Energy Consumption Data** - Household power (2M+ records)
  - Source: UCI ML Repository
  - Use: Time series forecasting

- **Bitcoin Historical Data** - Cryptocurrency prices
  - Source: Kaggle, CoinGecko API
  - Use: Time series prediction

- **Weather Data** - Historical weather from various sources
  - Source: NOAA, Kaggle
  - Use: Time series, regression

- **Traffic Volume Dataset** - Metro Interstate (48k records)
  - Source: UCI ML Repository
  - Use: Time series forecasting

---

## 🎬 Recommendation Systems

### Beginner
- **MovieLens 100K** - 100k ratings
  - Source: GroupLens
  - Use: Collaborative filtering

### Intermediate
- **MovieLens 25M** - 25M ratings from 162k users
  - Source: GroupLens
  - Use: Large-scale recommendation

- **Book Recommendation Dataset** - 1.1M ratings
  - Source: Kaggle
  - Use: Recommendation systems

- **Spotify Dataset** - Music recommendation features
  - Source: Kaggle
  - Use: Content-based filtering

---

## 🏆 Popular Kaggle Datasets

### Competition Datasets
1. **Titanic** - Survival prediction (beginner competition)
   - 891 training samples
   - Use: Binary classification

2. **House Prices** - Advanced regression
   - 1,460 samples, 79 features
   - Use: Feature engineering, regression

3. **Digit Recognizer** - MNIST on Kaggle
   - 42,000 training images
   - Use: Image classification

### Practice Datasets
4. **Iris Dataset** - Classic ML dataset (150 samples, 4 features)
   - Source: UCI, sklearn
   - Use: Multi-class classification

5. **Wine Quality** - Red and white wine (6,497 samples)
   - Source: UCI, Kaggle
   - Use: Classification/regression

6. **Used Car Price Prediction** - Vehicle pricing
   - Source: Kaggle
   - Use: Regression

---

## 🔥 Advanced/Specialized

### Large-Scale Datasets
- **Yelp Dataset** - 8M reviews, 200k images
  - Source: Yelp Dataset Challenge
  - Use: NLP, image analysis, recommendation

- **Airbnb Listings** - Pricing and occupancy prediction
  - Source: Inside Airbnb
  - Use: Regression, pricing optimization

### Business Analytics
- **Employee Attrition** - HR analytics dataset
  - Source: Kaggle
  - Use: Classification, feature importance

- **Hotel Booking Demand** - 119k reservations
  - Source: Kaggle
  - Use: Classification (cancellation prediction)

---

## 📚 How to Download Datasets

### From Kaggle
```python
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d <dataset-name>
```

### From UCI
```python
import pandas as pd

# Example: Heart Disease
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(url, header=None)
```

### From sklearn
```python
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

# Load built-in datasets
iris = load_iris()
cancer = load_breast_cancer()
wine = load_wine()
```

---

## 🎯 Recommended Learning Path

### Week 1-2: Beginner Datasets
- Iris (classification)
- Titanic (classification)
- House Prices (regression)

### Week 3-4: Intermediate Datasets
- Credit Card Fraud (imbalanced data)
- Customer Churn (business problem)
- IMDB Reviews (NLP)

### Week 5-6: Advanced Datasets
- Instacart (large-scale)
- MovieLens 25M (recommendation)
- Yelp Dataset (multi-modal)

---

## 💡 Tips for Working with Real Datasets

1. **Start Small** - Use subsets of large datasets initially
2. **Understand the Domain** - Read dataset descriptions carefully
3. **Handle Missing Data** - Real datasets have missing values
4. **Feature Engineering** - Critical for real-world performance
5. **Imbalanced Classes** - Common in real datasets
6. **Cross-Validation** - Essential for reliable evaluation
7. **Baseline Models** - Always start with simple models

---

## 📁 Datasets Folder Structure

Organize your datasets like this:
```
Machine Learning Hands-on/
├── datasets/
│   ├── titanic/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── house_prices/
│   ├── credit_fraud/
│   └── README.md (this file)
```

---

**Happy Dataset Hunting! 🔍**

*Remember: The best way to learn ML is by practicing with real-world data!*
