def predict_sentiment(model, vectorizer, text):

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)

    return "Positive" if prediction[0] == 1 else "Negative"