import streamlit as st
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
st.set_page_config(page_title='Suicide Detection AI', page_icon='🧠')

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
    # Classical ML components
    with open('tfidf.dill', 'rb') as f:
        vectorizer = dill.load(f)

    le = joblib.load('label_encoder.pkl')

    logistic_model = joblib.load('logistic_regression.pkl')
    nb_model = joblib.load('naive_bayes.pkl')
    svm_model = joblib.load('linear_svc.pkl')

    # Neural Network components
    bilstm_model = load_model('bilstm_model.keras')
    rnn_model = load_model('rnn_model.keras')

    with open('tokenizer_nn.pkl', 'rb') as f:
        tokenizer_nn = pickle.load(f)

    return vectorizer, le, logistic_model, nb_model, svm_model, bilstm_model, rnn_model, tokenizer_nn

try:
    vectorizer, le, logistic_model, nb_model, svm_model, bilstm_model, rnn_model, tokenizer_nn = load_artifacts()

    st.title('🧠 Suicide Detection Classifier')
    st.write('Enter text below to analyze the sentiment and risk level across multiple models.')

    user_input = st.text_area('Input Text', placeholder='How are you feeling today?')

    if st.button('Analyze'):
        if user_input.strip() == '':
            st.warning('Please enter some text first.')
        else:
            with st.spinner('Analyzing...'):
                # Preprocess
                tokens = preprocess_text(user_input)

                # Classical ML features
                X_classical = vectorizer.transform([tokens])

                # Neural Network features
                MAX_LEN = 200
                X_nn_seq = tokenizer_nn.texts_to_sequences([tokens])
                X_nn = pad_sequences(X_nn_seq, maxlen=MAX_LEN, padding='post', truncating='post')

                # Predictions
                lr_label = le.inverse_transform(logistic_model.predict(X_classical))[0]
                nb_label = le.inverse_transform(nb_model.predict(X_classical))[0]
                svm_label = le.inverse_transform(svm_model.predict(X_classical))[0]

                bilstm_prob = bilstm_model.predict(X_nn, verbose=0).flatten()[0]
                bilstm_label = le.inverse_transform([(bilstm_prob > 0.5).astype(int)])[0]

                rnn_prob = rnn_model.predict(X_nn, verbose=0).flatten()[0]
                rnn_label = le.inverse_transform([(rnn_prob > 0.5).astype(int)])[0]


                print(f"[DEBUG] BiLSTM   → Suicide: {bilstm_prob:.6f} | Non-Suicide: {1 - bilstm_prob:.6f}")
                print(f"[DEBUG] SimpleRNN → Suicide: {rnn_prob:.6f} | Non-Suicide: {1 - rnn_prob:.6f}")

            # --- Display Results ---
            st.subheader('📊 Model Predictions')

            # Classical Models
            st.markdown('**Classical ML Models**')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Logistic Regression', lr_label)
            with col2:
                st.metric('Naive Bayes', nb_label)
            with col3:
                st.metric('LinearSVC', svm_label)

            # Neural Models
            st.markdown('**Neural Network Models**')
            col4, col5 = st.columns(2)
            with col4:
                st.metric('BiLSTM', bilstm_label, delta=f'Confidence: {bilstm_prob:.2%}')
            with col5:
                st.metric('SimpleRNN', rnn_label, delta=f'Confidence: {rnn_prob:.2%}')

            # Summary verdict: flag risk if majority of models say so
            risk_votes = [
                lr_label, nb_label, svm_label, bilstm_label, rnn_label
            ].count('suicide')  # adjust label string to match your label encoder

            st.divider()
            st.subheader('🔍 Overall Assessment')
            if risk_votes >= 3:
                st.error(f'⚠️ High risk detected ({risk_votes}/5 models flagged). Please reach out for help.')
                st.info('**Malaysia Befrienders KL:** 03-7627 2929  |  **Crisis Text:** Text HOME to 741741')
            else:
                st.success(f'✅ Low risk detected ({risk_votes}/5 models flagged).')

except Exception as e:
    st.error(f'Error loading models: {e}')
    st.info('Ensure all saved artifacts (.pkl, .dill, .keras) are in the current directory.')