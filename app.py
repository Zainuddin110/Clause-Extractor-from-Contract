import streamlit as st
import fitz  # PyMuPDF
import joblib
import spacy
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

@st.cache_resource
def load_model_and_vectorizer():
    clf = joblib.load('clause_classifier_rf.joblib')
    vectorizer = joblib.load('tfidf_vectorizer_rf.joblib')
    return clf, vectorizer

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

clf, vectorizer = load_model_and_vectorizer()
nlp = load_spacy()

st.title("Legal Contract Clause Extraction")

uploaded_file = st.file_uploader("Upload Contract File (.txt or .pdf)", type=['txt', 'pdf'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        contract_text = ""
        for page in pdf_doc:
            contract_text += page.get_text()
    else:
        # Plain text file
        contract_text = uploaded_file.read().decode('utf-8')

    st.subheader("Contract Text Preview")
    st.text_area("", contract_text, height=300)

    # Use spaCy for sentence splitting
    doc = nlp(contract_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    st.write(f"Total sentences extracted: {len(sentences)}")

    # Predict clause type
    predictions = []
    for sent in sentences:
        cleaned = clean_text(sent)
        lemmatized = " ".join([token.lemma_ for token in nlp(cleaned)])
        vec_sent = vectorizer.transform([lemmatized])
        pred = clf.predict(vec_sent)[0]
        predictions.append(pred)

    st.subheader("Clause Predictions")
    for i, (sentence, pred) in enumerate(zip(sentences, predictions), 1):
        st.markdown(f"**Sentence {i}:** {sentence}")
        st.markdown(f"**Predicted Clause Type:** {pred}")
        st.markdown("---")