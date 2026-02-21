import pickle


def predict():
    # Load model
    with open("model/spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load vectorizer
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    msg = input("Enter message: ")

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        print("Spam")
    else:
        print("Not Spam")


if __name__ == "__main__":
    predict()