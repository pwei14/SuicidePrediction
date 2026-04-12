import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import dill
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs, LangDetectException

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)

# Page configuration
st.set_page_config(page_title='Suicide Detection AI', page_icon='🧠', layout='wide')

# Preprocessing setup
stop_words = set(stopwords.words('english'))
extra_stops = {'im', 'ive', 'dont', 'cant', 'didnt', 'wasnt', 'isnt', 'id', 'ill', 'youre', 'theyre', 'werent', 'hasnt', 'havent'}
stop_words.update(extra_stops)
lemmatizer = WordNetLemmatizer()

# Load English word dictionary once
english_dict = set(w.lower() for w in nltk.corpus.words.words())

# Malay blocklist
malay_blocklist = {
    'saya', 'nak', 'aku', 'kau', 'dia', 'kami', 'kita', 'mereka',
    'tidak', 'tak', 'ada', 'dan', 'atau', 'dengan', 'untuk', 'dari',
    'ini', 'itu', 'yang', 'sudah', 'akan', 'boleh', 'mahu', 'ingin',
    'rasa', 'hidup', 'mati', 'sakit', 'sedih', 'senang', 'susah',
    'suka', 'benci', 'cinta', 'rindu', 'marah', 'takut', 'harap',
    'dah', 'lah', 'pun', 'je', 'juga', 'masih', 'lagi', 'sangat',
    'sekali', 'semua', 'pergi', 'datang', 'makan', 'minum', 'tidur',
    'bangun', 'kerja', 'rumah', 'sekolah', 'keluarga', 'kawan', 'orang',
    'macam', 'camne', 'camana', 'kenapa', 'siapa', 'mana', 'bila',
    'berapa', 'bagaimana', 'mengapa', 'sebab', 'kerana', 'supaya',
    'kalau', 'jika', 'walaupun', 'tetapi', 'tapi', 'namun', 'sungguh'
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def is_non_english(text):
    stripped = text.strip()
    if not all(ord(c) < 128 for c in stripped if not c.isspace()):
        return True
    words = [w.lower() for w in stripped.split() if w.isalpha()]
    if len(words) == 0:
        return False
    for w in words:
        if w in malay_blocklist:
            return True
    matched = sum(1 for w in words if w in english_dict)
    ratio = matched / len(words)
    if ratio < 0.6:
        return True
    if len(words) >= 7:
        try:
            langs = detect_langs(stripped)
            top = langs[0]
            if top.lang != 'en' and top.prob > 0.95:
                return True
        except LangDetectException:
            pass
    return False

def render_gauge(risk_ratio):
    """Render a circular gauge using Plotly-like HTML/JS via streamlit components."""
    percent = int(round(risk_ratio * 100))
    if risk_ratio < 0.34:
        color = '#639922'
        level = 'Low Risk'
    elif risk_ratio < 0.67:
        color = '#EF9F27'
        level = 'Medium Risk'
    else:
        color = '#E24B4A'
        level = 'High Risk'

    gauge_html = f"""
    <div style="display:flex; align-items:center; gap:32px; background:white; border:0.5px solid #e0e0e0;
                border-radius:12px; padding:20px 28px; margin-top:8px;">
      <div style="position:relative; width:140px; height:80px; flex-shrink:0;">
        <canvas id="gaugeCanvas" width="140" height="80"></canvas>
      </div>
      <div>
        <div style="font-size:26px; font-weight:500; color:{color}; margin-bottom:4px;">{level}</div>
        <div style="font-size:13px; color:#888; margin-bottom:10px;">Overall risk score: {percent}%</div>
      </div>
    </div>
    <script>
    (function() {{
      var canvas = document.getElementById('gaugeCanvas');
      var ctx = canvas.getContext('2d');
      var cx = 70, cy = 76, r = 56;
      var zones = [
        {{ end: 0.33, color: '#639922' }},
        {{ end: 0.66, color: '#EF9F27' }},
        {{ end: 1.0,  color: '#E24B4A' }}
      ];
      var riskRatio = {risk_ratio};

      zones.forEach(function(z, i) {{
        var prev = i === 0 ? 0 : zones[i-1].end;
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI + prev * Math.PI, Math.PI + z.end * Math.PI);
        ctx.lineWidth = 13;
        ctx.strokeStyle = 'rgba(0,0,0,0.08)';
        ctx.lineCap = 'butt';
        ctx.stroke();
      }});

      var filled = 0;
      zones.forEach(function(z, i) {{
        var prev = i === 0 ? 0 : zones[i-1].end;
        var end = Math.min(riskRatio, z.end);
        if (end > prev) {{
          ctx.beginPath();
          ctx.arc(cx, cy, r, Math.PI + prev * Math.PI, Math.PI + end * Math.PI);
          ctx.lineWidth = 13;
          ctx.strokeStyle = z.color;
          ctx.lineCap = 'butt';
          ctx.stroke();
        }}
      }});

      var needleAngle = Math.PI + riskRatio * Math.PI;
      var nx = cx + (r - 8) * Math.cos(needleAngle);
      var ny = cy + (r - 8) * Math.sin(needleAngle);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#444';
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#444';
      ctx.fill();

      ctx.font = '500 12px sans-serif';
      ctx.fillStyle = '#666';
      ctx.textAlign = 'center';
      ctx.fillText('{percent}%', cx, cy - 16);
    }})();
    </script>
    """
    st.components.v1.html(gauge_html, height=120)

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

# Color card CSS
st.markdown("""
<style>
.result-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 6px;
}
.card-suicide {
    background-color: #FCEBEB;
    border: 1px solid #F09595;
}
.card-non-suicide {
    background-color: #EAF3DE;
    border: 1px solid #97C459;
}
.card-model-name {
    font-size: 13px;
    color: #555;
    margin-bottom: 4px;
}
.card-label-suicide {
    font-size: 22px;
    font-weight: 600;
    color: #A32D2D;
}
.card-label-non-suicide {
    font-size: 22px;
    font-weight: 600;
    color: #3B6D11;
}
.card-conf {
    font-size: 12px;
    margin-top: 6px;
    color: #666;
}
.conf-bar-bg {
    background: rgba(0,0,0,0.08);
    border-radius: 99px;
    height: 5px;
    margin-top: 4px;
    overflow: hidden;
}
.conf-bar-fill-suicide {
    height: 100%;
    border-radius: 99px;
    background: #E24B4A;
}
.conf-bar-fill-non-suicide {
    height: 100%;
    border-radius: 99px;
    background: #639922;
}
</style>
""", unsafe_allow_html=True)

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
                if is_non_english(user_input):
                    st.error('🌐 Please input text in English only and try again.')
                    st.stop()

                # --- Progress Bar Animation ---
                st.subheader('⏳ Analyzing...')
                progress_placeholder = st.empty()

                model_steps = []
                if 'Logistic Regression' in selected_models:
                    model_steps.append('Logistic Regression')
                if 'Naive Bayes' in selected_models:
                    model_steps.append('Naive Bayes')
                if 'LinearSVC' in selected_models:
                    model_steps.append('LinearSVC')
                if 'BiLSTM' in selected_models:
                    model_steps.append('BiLSTM')
                if 'SimpleRNN' in selected_models:
                    model_steps.append('SimpleRNN')

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
                completed = []

                for step in model_steps:
                    with progress_placeholder.container():
                        for s in model_steps:
                            if s in completed:
                                st.progress(1.0, text=f'✅ {s} — done')
                            elif s == step:
                                st.progress(0.6, text=f'⏳ {s} — running...')
                            else:
                                st.progress(0.0, text=f'🔲 {s} — waiting')

                    if step == 'Logistic Regression':
                        results['Logistic Regression'] = {
                            'label': le.inverse_transform(logistic_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    elif step == 'Naive Bayes':
                        results['Naive Bayes'] = {
                            'label': le.inverse_transform(nb_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    elif step == 'LinearSVC':
                        results['LinearSVC'] = {
                            'label': le.inverse_transform(svm_model.predict(X_classical))[0],
                            'type': 'classical'
                        }
                    elif step == 'BiLSTM':
                        bilstm_prob = bilstm_model.predict(X_nn, verbose=0).flatten()[0]
                        bilstm_label = le.inverse_transform([(bilstm_prob > 0.5).astype(int)])[0]
                        results['BiLSTM'] = {
                            'label': bilstm_label,
                            'prob': float(bilstm_prob),
                            'type': 'nn'
                        }
                    elif step == 'SimpleRNN':
                        rnn_prob = rnn_model.predict(X_nn, verbose=0).flatten()[0]
                        rnn_label = le.inverse_transform([(rnn_prob > 0.5).astype(int)])[0]
                        results['SimpleRNN'] = {
                            'label': rnn_label,
                            'prob': float(rnn_prob),
                            'type': 'nn'
                        }

                    completed.append(step)
                    time.sleep(0.3)

                # Mark all done
                with progress_placeholder.container():
                    for s in model_steps:
                        st.progress(1.0, text=f'✅ {s} — done')

                time.sleep(0.4)
                progress_placeholder.empty()

                # --- Display Results ---
                st.subheader('📊 Model Predictions')

                classical_results = {k: v for k, v in results.items() if v['type'] == 'classical'}
                nn_results = {k: v for k, v in results.items() if v['type'] == 'nn'}

                if classical_results:
                    st.markdown('**Classical ML Models**')
                    cols = st.columns(len(classical_results))
                    for col, (name, res) in zip(cols, classical_results.items()):
                        label = res['label']
                        card_class = 'card-suicide' if label == 'suicide' else 'card-non-suicide'
                        label_class = 'card-label-suicide' if label == 'suicide' else 'card-label-non-suicide'
                        with col:
                            st.markdown(f"""
                            <div class="result-card {card_class}">
                                <div class="card-model-name">{name}</div>
                                <div class="{label_class}">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                if nn_results:
                    st.markdown('**Neural Network Models**')
                    cols = st.columns(len(nn_results))
                    for col, (name, res) in zip(cols, nn_results.items()):
                        label = res['label']
                        prob = res['prob']
                        card_class = 'card-suicide' if label == 'suicide' else 'card-non-suicide'
                        label_class = 'card-label-suicide' if label == 'suicide' else 'card-label-non-suicide'
                        fill_class = 'conf-bar-fill-suicide' if label == 'suicide' else 'conf-bar-fill-non-suicide'
                        bar_width = int(prob * 100)
                        with col:
                            st.markdown(f"""
                            <div class="result-card {card_class}">
                                <div class="card-model-name">{name}</div>
                                <div class="{label_class}">{label}</div>
                                <div class="card-conf">Suicide probability: {prob:.2%}</div>
                                <div class="conf-bar-bg">
                                    <div class="{fill_class}" style="width:{bar_width}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # --- Gauge + Overall Assessment ---
                all_labels = [v['label'] for v in results.values()]
                risk_votes = all_labels.count('suicide')
                total = len(all_labels)
                risk_ratio = risk_votes / total

                st.divider()
                st.subheader('🔍 Overall Assessment')

                render_gauge(risk_ratio)

                st.markdown(f"**{risk_votes}/{total} models** flagged as suicide risk.")

                if risk_votes >= (total / 2 + 1):
                    st.error(f'⚠️ High risk detected. Please reach out for help.')
                    st.info('**Malaysia Befrienders KL:** 03-7627 2929  |  **Crisis Text:** Text HOME to 741741')
                else:
                    st.success(f'✅ Low risk detected.')

except Exception as e:
    st.error(f'Error loading models: {e}')
    st.info('Ensure all saved artifacts (.pkl, .dill, .keras) are in the current directory.')