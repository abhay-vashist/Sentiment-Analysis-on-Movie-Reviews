from preprocess import clean_text

def predict_sentiment(review, model, tfidf):
    cleaned = clean_text(review)
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😞"
