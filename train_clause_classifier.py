import pandas as pd
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load spaCy for lemmatization
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def main():
    # Load your dataset CSV
    df = pd.read_csv('legal_docs.csv')

    # Clean and lemmatize text
    df['cleaned_text'] = df['clause_text'].apply(clean_text)
    df['lemmatized'] = df['cleaned_text'].apply(lemmatize_text)

    # Prepare features and labels
    X = df['lemmatized']
    y = df['clause_type']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.4)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=3110, ngram_range=(1,3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(clf, 'clause_classifier_rf.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer_rf.joblib')

if __name__ == '__main__':
    main()