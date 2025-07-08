import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize stemmer and stopword set
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    stemmed = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(stemmed)

# Predict sentiment
def predict_tweet_sentiment(tweet):
    cleaned = preprocess(tweet)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Positive 😊" if pred == 1 else "Negative 😡"

# Example usage
if __name__ == "__main__":
    tweet = input("Enter a tweet: ")
    print("Prediction:", predict_tweet_sentiment(tweet))
