import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text
from utils import evaluate_model

def train_models(dataset_path="IMDB_Dataset.csv"):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Preprocess text
    df['cleaned_review'] = df['review'].apply(clean_text)
    y = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

    # Feature extraction
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_review']).toarray()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    evaluate_model("Naive Bayes", y_test, y_pred_nb)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    evaluate_model("Logistic Regression", y_test, y_pred_lr)

    return nb_model, lr_model, tfidf
