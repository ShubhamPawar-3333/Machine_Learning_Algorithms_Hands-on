# Naive Bayes 📊

## Overview
Naive Bayes is a family of probabilistic classifiers based on **Bayes' Theorem** with the "naive" assumption of independence between features. Despite this simplification, it works surprisingly well for many real-world applications.

## Mathematical Foundation

### Bayes' Theorem
```
P(A|B) = [P(B|A) × P(A)] / P(B)
```

### For Classification
```
P(Class|Features) = [P(Features|Class) × P(Class)] / P(Features)
```

We predict the class with highest posterior probability:
```
Class = argmax P(Class|Features)
```

### Naive Assumption
Features are **conditionally independent** given the class:
```
P(x₁, x₂, ..., xₙ|Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class)
```

## Types of Naive Bayes

### 1. Gaussian Naive Bayes
- **Use Case**: Continuous features
- **Assumption**: Features follow normal (Gaussian) distribution
- **Example**: Height, weight, temperature

### 2. Multinomial Naive Bayes
- **Use Case**: Discrete counts (word frequencies)
- **Assumption**: Features represent frequencies
- **Example**: Text classification, document categorization

### 3. Bernoulli Naive Bayes
- **Use Case**: Binary/Boolean features
- **Assumption**: Features are binary (0 or 1)
- **Example**: Word presence/absence in text

### 4. Complement Naive Bayes
- **Use Case**: Imbalanced datasets
- **Improvement**: Works better than Multinomial for imbalanced data

## Key Concepts

### Prior Probability
```
P(Class) = Count(Class) / Total_Samples
```

### Likelihood
```
P(Feature|Class) = Count(Feature in Class) / Count(Class)
```

### Laplace Smoothing
Prevents zero probability problem:
```
P(Feature|Class) = [Count(Feature in Class) + α] / [Count(Class) + α × n_features]
```
Where α = 1 (default smoothing parameter)

## Advantages
✅ Fast training and prediction  
✅ Works well with small datasets  
✅ Handles high-dimensional data well  
✅ Not sensitive to irrelevant features  
✅ Performs well with categorical data  
✅ Provides probabilistic predictions  
✅ Simple and easy to implement  

## Disadvantages
❌ Assumes feature independence (rarely true)  
❌ Zero probability problem (solved by smoothing)  
❌ Not suitable for regression  
❌ Can be outperformed by more complex models  
❌ Sensitive to how features are presented  

## Use Cases
- 📧 **Spam Detection** - Email classification (spam/not spam)
- 📝 **Sentiment Analysis** - Positive/negative reviews
- 📰 **News Categorization** - Topic classification
- 🏥 **Medical Diagnosis** - Disease prediction
- 🎬 **Recommendation Systems** - User preferences
- 🔍 **Search Engines** - Document relevance
- 📱 **Real-time Prediction** - Fast classification needed

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Log Loss (for probability calibration)

## Real-World Datasets
1. **SMS Spam Collection** - Spam detection
2. **20 Newsgroups** - Text classification
3. **IMDB Reviews** - Sentiment analysis
4. **Iris Dataset** - Multi-class classification
5. **Titanic Dataset** - Survival prediction
6. **Email Classification** - Category prediction

## Text Classification Example

### Workflow
1. **Preprocessing**: Tokenization, lowercasing, removing stopwords
2. **Vectorization**: Convert text to numerical features
   - Bag of Words
   - TF-IDF (Term Frequency-Inverse Document Frequency)
3. **Training**: Calculate probabilities
4. **Prediction**: Apply Bayes' theorem

## Handling Zero Probability

### Problem
If a feature never appears with a class in training:
```
P(Feature|Class) = 0
→ P(Class|Features) = 0 (entire probability becomes zero!)
```

### Solution: Laplace Smoothing
Add small constant (α = 1) to all counts

## Comparison with Other Algorithms

| Aspect | Naive Bayes | Logistic Regression | SVM |
|--------|-------------|---------------------|-----|
| Speed | Very Fast | Fast | Slow |
| Accuracy | Good | Better | Best |
| Interpretability | High | High | Low |
| Feature Independence | Required | Not Required | Not Required |
| Probabilistic | Yes | Yes | No (by default) |

## When to Use Naive Bayes

### ✅ Good For:
- Text classification
- Real-time predictions
- Multi-class problems
- Small to medium datasets
- High-dimensional data
- Baseline model

### ❌ Not Good For:
- Features are highly correlated
- Need high accuracy
- Regression problems
- Complex decision boundaries

## Hyperparameters

### Gaussian Naive Bayes
- **var_smoothing** - Portion of largest variance added to variances

### Multinomial/Bernoulli Naive Bayes
- **alpha** - Smoothing parameter (default: 1.0)
- **fit_prior** - Whether to learn class prior probabilities

## Files in This Folder
- `naive_bayes_tutorial.ipynb` - Interactive Jupyter notebook
- `naive_bayes.py` - Python implementation from scratch
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- [Text Classification with Naive Bayes](https://www.youtube.com/watch?v=l3dZ6ZNFjo0)
- [Stanford NLP Course](https://web.stanford.edu/~jurafsky/slp3/)

## Next Steps
1. Open `naive_bayes_tutorial.ipynb` in Jupyter
2. Understand Bayes' theorem and conditional probability
3. Practice with text classification (spam detection)
4. Compare different types of Naive Bayes
5. Experiment with Laplace smoothing
6. Build a sentiment analyzer

---
**Happy Learning! 🚀**
