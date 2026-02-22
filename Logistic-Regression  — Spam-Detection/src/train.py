import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train():
    # Load dataset
    data = pd.read_csv("data/spam.csv", encoding='latin-1')

    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    # Convert labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'],
        data['label'],
        test_size=0.2,
        random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    with open("model/spam_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model saved successfully!")


if __name__ == "__main__":
    train()