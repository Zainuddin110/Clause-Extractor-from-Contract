import os
import streamlit as st
import joblib
import spacy
import re
import gdown
import fitz  # PyMuPDF for PDF text extraction

# URLs for hosted models on Google Drive (replace FILE_ID with your actual IDs)
MODEL_URL = "https://drive.google.com/uc?id=1loKGKBPyenJehnWw88tonhhOK0a0M9xA"
VECTORIZER_URL = "https://drive.google.com/uc?id=1JI17Jg2v3WdjsxhOjJ9KvBCWQhfrjUEd"

def download_model_files():
    if not os.path.exists("clause_classifier.joblib"):
        st.info("Downloading model file...")
        gdown.download(MODEL_URL, "clause_classifier.joblib", quiet=False)
    if not os.path.exists("tfidf_vectorizer.joblib"):
        st.info("Downloading vectorizer file...")
        gdown.download(VECTORIZER_URL, "tfidf_vectorizer.joblib", quiet=False)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

@st.cache_resource
def load_spacy():
    import spacy
    import os
    if not spacy.util.is_package("en_core_web_sm"):
        os.system("python -m spacy download en_core_web_sm")
    return spacy.load("en_core_web_sm")

def main():
    st.title("Legal Contract Clause Extraction (Streamlit Cloud Ready)")

    download_model_files()
    clf = joblib.load("clause_classifier.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    nlp = load_spacy()

    uploaded_file = st.file_uploader(
        "Upload Contract File (.txt or .pdf)", type=['txt', 'pdf']
    )

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            contract_text = ""
            for page in pdf_doc:
                contract_text += page.get_text()
        else:
            contract_text = uploaded_file.read().decode('utf-8')

        st.subheader("Contract Text Preview")
        st.text_area("", contract_text, height=300)

        doc = nlp(contract_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        st.write(f"Total sentences extracted: {len(sentences)}")

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

if __name__ == "__main__":
    main()


