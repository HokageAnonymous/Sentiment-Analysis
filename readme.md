# Sentiment Analysis using Machine Learning

This project performs **Sentiment Analysis** on 1.6 million tweets from the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The tweets are preprocessed, vectorized using TF-IDF, and classified using a Logistic Regression model.

---

## 📊 Dataset Overview

- **Source**: Kaggle - Sentiment140
- **Records**: 1,600,000 tweets
- **Classes**:
  - `0` = Negative sentiment
  - `4` = Positive sentiment (converted to `1`)

Dataset fields:

- `target`: Sentiment label (0 or 4)
- `id`: Tweet ID
- `date`: Timestamp
- `flag`: Query used
- `user`: Username
- `text`: Actual tweet

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- NLTK (for stopwords & stemming)
- Scikit-learn (TF-IDF, Logistic Regression)
- Pickle (for saving model)

---

## 🧹 Data Preprocessing

- Removed punctuation & non-alphabetic characters
- Converted text to lowercase
- Removed English stopwords
- Applied **Porter Stemming**
- Vectorized with **TF-IDF** (max 100,000 features)

---

## 🤖 Model Details

- **Algorithm**: Logistic Regression
- **Vectorizer**: TfidfVectorizer
- **Split**: 80% Train / 20% Test

### 📈 Accuracy

- **Training**: \~79.8%
- **Testing**: \~77.6%

---

## 📁 Project Structure

```
Sentiment-Analysis/
├── training.1600000.processed.noemoticon.csv
├── sentiment_model.pkl            # Trained model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── sentiment_analysis.py          # Main training & preprocessing code
├── predict_example.py             # Sample script to load and predict
└── README.md                      # Project summary (this file)
```

---

## 🔍 Example Prediction

```python
from predict_example import predict_tweet_sentiment

print(predict_tweet_sentiment("I'm feeling amazing today!"))  # Output: Positive 😊
```

---

## 🧠 Future Improvements

- Replace Logistic Regression with BERT (using Hugging Face Transformers)
- Add web or mobile interface for real-time predictions
- Add neutral class (multi-class classification)
- Use `Pipeline` to automate preprocessing & classification

---

## ✍️ Author

- Rituraj Kanchan
- MS AICTE Internship Project (2025)

---

## 📜 License

This project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) under Kaggle's data terms.

