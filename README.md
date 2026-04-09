# Naive Bayes Spam Detection using Machine Learning

## Project Overview

This project demonstrates how a **Naive Bayes Classification algorithm** can be used to classify SMS messages as **Spam** or **Ham (Not Spam)** using Machine Learning.

The model is trained on a labelled SMS dataset and predicts whether a new incoming message belongs to:

* **Spam** → unwanted / promotional / suspicious message
* **Ham** → genuine normal message

The project also includes:

* Word-level spam/ham probability analysis
* Full statement probability prediction
* Confusion matrix visualization
* Accuracy evaluation
* Top strongest spam and ham words

---

# 1. Background: Machine Learning Classification

Machine Learning algorithms are broadly divided into different learning paradigms depending on the availability of labelled data.

---

## 1.1 Supervised Learning

Supervised Learning is used when the dataset contains:

* **Input features (X)**
* **Known output labels (Y)**

The model learns the relationship between input and output using labelled examples.

### Example

| Message            | Label |
| ------------------ | ----- |
| Win free prize now | Spam  |
| Meet me tomorrow   | Ham   |

Here:

* input = message text
* output = spam / ham

The model learns patterns from known examples and predicts unseen future data.

---

## 1.2 Unsupervised Learning

Unsupervised Learning is used when:

* data has **no labels**
* model discovers hidden patterns automatically

### Example

Grouping customers based on similar buying behaviour.

Common tasks:

* clustering
* dimensionality reduction
* association rule mining

---

## 1.3 Other Learning Categories

### Semi-Supervised Learning

Uses:

* small labelled data
* large unlabelled data

### Reinforcement Learning

Agent learns through:

* reward
* penalty
* repeated actions

Used in:

* robotics
* game AI
* autonomous systems

---

# 2. Why Naive Bayes belongs to Supervised Learning

Naive Bayes requires labelled training data.

Our dataset already contains labels:

* ham
* spam

Therefore it learns from known examples before predicting new text messages.

---

# 3. Why Naive Bayes was chosen for this project

Naive Bayes is highly suitable for text classification because:

* fast training
* efficient for large text vocabulary
* performs very well on word-frequency features
* probabilistic outputs are easy to interpret

It is widely used in:

* email spam filtering
* sentiment analysis
* fraud message detection
* document classification

---

# 4. Problem Statement

To build a machine learning model that can automatically classify SMS messages into:

* Spam
* Ham

and calculate:

* spam probability
* ham probability

for:

* individual words
* full user-entered statements

---

# 5. Dataset Used

Dataset used:

SMS Spam Collection Dataset

Dataset source:

https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

---

## Dataset Structure

| label | message                     |
| ----- | --------------------------- |
| ham   | Go until jurong point       |
| spam  | Free entry in 2 a wkly comp |

---

## Label Meaning

### Ham

Normal legitimate message.

### Spam

Promotional / suspicious / unsolicited message.

---

# 6. Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

# 7. Project Workflow

---

## Step 1: Dataset Loading

Dataset is loaded using pandas.

```python
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])
```

---

## Step 2: Data Exploration

Checked:

* dataset size
* null values
* label distribution

Output:

* 5572 total messages

---

## Step 3: Label Encoding

Labels converted into numeric form:

* ham = 0
* spam = 1

```python
df['label_num'] = df.label.map({'ham':0, 'spam':1})
```

---

## Step 4: Train-Test Split

Dataset divided into:

* 80% training
* 20% testing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 5: Text Vectorization

Text converted into numerical word-frequency matrix.

```python
vect = CountVectorizer(
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
)
```

---

## Why Vectorization is needed

Machine learning models cannot directly understand text.

Words must be converted into numerical form before training.

---

## Step 6: Model Training

Used:

Multinomial Naive Bayes

```python
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
```

---

# 8. Model Evaluation

---

## Accuracy Obtained

**98.74%**

---

## Classification Report

Shows:

* precision
* recall
* f1-score

---

## Confusion Matrix

Interprets:

* true ham
* true spam
* false positives
* false negatives

Example:

| Actual | Predicted |
| ------ | --------- |
| Ham    | Ham       |
| Spam   | Spam      |

---

# 9. Manual Testing of Messages

Example:

```python
predict_spam("Congratulations! You've won a gift card")
```

Output:

```text
SPAM
```

---

# 10. Full Statement Probability Prediction

Added feature where user enters full statement.

Model returns:

* predicted label
* ham probability
* spam probability

Example:

Input:

```text
Claim your free reward now
```

Output:

```text
Ham Probability : 2.14%
Spam Probability: 97.86%
```

---

## Code Used

```python
def predict_message_with_probability(message):
    message_dtm = vect.transform([message])

    prediction = nb.predict(message_dtm)[0]
    probabilities = nb.predict_proba(message_dtm)[0]

    print("Message:", message)
    print("Prediction:", "SPAM" if prediction == 1 else "HAM")
    print(f"Ham Probability : {probabilities[0]*100:.2f}%")
    print(f"Spam Probability: {probabilities[1]*100:.2f}%")
```

---

# 11. Word-Level Probability Analysis

Each learned word has:

* ham probability
* spam probability

Example:

| Word    | Ham % | Spam % |
| ------- | ----- | ------ |
| free    | 10%   | 90%    |
| meeting | 94%   | 6%     |

---

# 12. Strongest Spam Words Found

Examples:

* claim
* prize
* guaranteed
* ringtone
* awarded

These words strongly indicate spam messages.

---

# 13. Strongest Ham Words Found

Examples:

* later
* ask
* doing
* said
* meeting

These words strongly indicate normal communication.

---

# 14. Visualizations Included

Project includes:

* Spam vs Ham distribution graph
* Message length histogram
* Confusion matrix heatmap
* Top spam words bar chart
* Top ham words bar chart

---

# 15. Why Naive Bayes Works Well Here

Naive Bayes assumes word independence.

Even though words are not fully independent in reality, it performs extremely well because spam words usually carry strong independent signals.

Example:

free + prize + claim

Each strongly increases spam probability.

---

# 16. Real World Applications

Naive Bayes is used in:

* Gmail spam filters
* SMS fraud detection
* phishing detection
* customer review classification
* sentiment analysis

---

# 17. Conclusion

This project demonstrates a complete machine learning pipeline:

* labelled text data
* preprocessing
* vectorization
* supervised learning
* probability prediction
* interpretable results

Naive Bayes gives high performance with very low computational cost.

---

# 18. Future Improvements

Possible future extensions:

* TF-IDF Vectorization
* Logistic Regression comparison
* Word Cloud visualization
* False prediction analysis
* Web deployment

---

# 19. Repository Structure

```text
├── README.md
├── notebook.ipynb
├── dataset/
└── screenshots/
```

---

# 20. Author

Prepared as part of Machine Learning demonstration project using Naive Bayes Classification.
