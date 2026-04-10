import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import dill
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Page configuration
st.set_page_config(page_title='Suicide Detection AI', page_icon='🧠', layout='wide')

# Preprocessing setup
stop_words = set(stopwords.words('english'))
extra_stops = {'im', 'ive', 'dont', 'cant', 'didnt', 'wasnt', 'isnt', 'id', 'ill', 'youre', 'theyre', 'werent', 'hasnt', 'havent'}
stop_words.update(extra_stops)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Full preprocessing pipeline matching test_models.py"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

@st.cache_resource
def load_artifacts():
    with open('tfidf.dill', 'rb') as f:
        vectorizer = dill.load(f)
    le = joblib.load('label_encoder.pkl')
    logistic_model = joblib.load('logistic_regression.pkl')
    nb_model = joblib.load('naive_bayes.pkl')
    svm_model = joblib.load('linear_svc.pkl')
    bilstm_model = load_model('bilstm_model.keras')
    rnn_model = load_model('rnn_model.keras')
    with open('tokenizer_nn.pkl', 'rb') as f:
        tokenizer_nn = pickle.load(f)
    return vectorizer, le, logistic_model, nb_model, svm_model, bilstm_model, rnn_model, tokenizer_nn

ALL_MODELS = [
    'Logistic Regression',
    'Naive Bayes',
    'LinearSVC',
    'BiLSTM',
    'SimpleRNN',
]

try:
    vectorizer, le, logistic_model, nb_model, svm_model, bilstm_model, rnn_model, tokenizer_nn = load_artifacts()

    # --- Sidebar ---
    with st.sidebar:
        st.title('🧠 Suicide Detection Classifier')
        st.write(
            'Analyze text for suicide risk using multiple machine learning models. '
            'Select which models to run and enter your text to get started.'
        )
        st.divider()

        st.subheader('Model Selection')
        st.write('Choose which models to run:')
        selected_models = st.multiselect(
            label='',
            options=ALL_MODELS,
            default=ALL_MODELS,
            label_visibility='collapsed'
        )

        st.divider()
        with st.expander('ℹ️ Model Info'):
            st.markdown('''
| Model | Type |
|---|---|
| Logistic Regression | Classical ML |
| Naive Bayes | Classical ML |
| LinearSVC | Classical ML |
| BiLSTM | Neural Network |
| SimpleRNN | Neural Network |
''')

    # --- Main Page ---
    st.title('🧠 Suicide Detection Classifier')
    st.write('Enter text below to analyze the risk level across your selected models.')

    if not selected_models:
        st.warning('Please select at least one model from the sidebar.')
    else:
        user_input = st.text_area('Input Text', placeholder='How are you feeling today?')

        if st.button('Analyze'):
            if user_input.strip() == '':
                st.warning('Please enter some text first.')
            else:
                with st.spinner('Analyzing...'):
                    tokens = preprocess_text(user_input)

                    needs_classical = any(m in selected_models for m in ['Logistic Regression', 'Naive Bayes', 'LinearSVC'])
                    needs_nn = any(m in selected_models for m in ['BiLSTM', 'SimpleRNN'])

                    if needs_classical:
                        X_classical = vectorizer.transform([tokens])

                    if needs_nn:
                        MAX_LEN = 200
                        X_nn_seq = tokenizer_nn.texts_to_sequences([tokens])
                        X_nn = pad_sequences(X_nn_seq, maxlen=MAX_LEN, padding='post', truncating='post')

                    results = {}

                    if 'Logistic Regression' in selected_models:
                        results['Logistic Regression'] = {
                            'label': le.inverse_transform(logistic_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    if 'Naive Bayes' in selected_models:
                        results['Naive Bayes'] = {
                            'label': le.inverse_transform(nb_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    if 'LinearSVC' in selected_models:
                        results['LinearSVC'] = {
                            'label': le.inverse_transform(svm_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    if 'BiLSTM' in selected_models:
                        bilstm_prob = bilstm_model.predict(X_nn, verbose=0).flatten()[0]
                        bilstm_label = le.inverse_transform([(bilstm_prob > 0.5).astype(int)])[0]
                        print(f"[DEBUG] BiLSTM   → Suicide: {bilstm_prob:.6f} | Non-Suicide: {1 - bilstm_prob:.6f}")
                        results['BiLSTM'] = {
                            'label': bilstm_label,
                            'prob': bilstm_prob,
                            'type': 'nn'
                        }
                    if 'SimpleRNN' in selected_models:
                        rnn_prob = rnn_model.predict(X_nn, verbose=0).flatten()[0]
                        rnn_label = le.inverse_transform([(rnn_prob > 0.5).astype(int)])[0]
                        print(f"[DEBUG] SimpleRNN → Suicide: {rnn_prob:.6f} | Non-Suicide: {1 - rnn_prob:.6f}")
                        results['SimpleRNN'] = {
                            'label': rnn_label,
                            'prob': rnn_prob,
                            'type': 'nn'
                        }

                # --- Display Results ---
                st.subheader('📊 Model Predictions')

                classical_results = {k: v for k, v in results.items() if v['type'] == 'classical'}
                nn_results = {k: v for k, v in results.items() if v['type'] == 'nn'}

                if classical_results:
                    st.markdown('**Classical ML Models**')
                    cols = st.columns(len(classical_results))
                    for col, (name, res) in zip(cols, classical_results.items()):
                        with col:
                            st.metric(name, res['label'])

                if nn_results:
                    st.markdown('**Neural Network Models**')
                    cols = st.columns(len(nn_results))
                    for col, (name, res) in zip(cols, nn_results.items()):
                        with col:
                            st.metric(name, res['label'], delta=f"Confidence: {res['prob']:.2%}")

                # Majority vote verdict
                all_labels = [v['label'] for v in results.values()]
                risk_votes = all_labels.count('suicide')  # adjust to match your label encoder
                total = len(all_labels)

                st.divider()
                st.subheader('🔍 Overall Assessment')
                if risk_votes >= (total / 2 + 1):
                    st.error(f'⚠️ High risk detected ({risk_votes}/{total} models flagged). Please reach out for help.')
                    st.info('**Malaysia Befrienders KL:** 03-7627 2929  |  **Crisis Text:** Text HOME to 741741')
                else:
                    st.success(f'✅ Low risk detected ({risk_votes}/{total} models flagged).')

except Exception as e:
    st.error(f'Error loading models: {e}')
    st.info('Ensure all saved artifacts (.pkl, .dill, .keras) are in the current directory.')