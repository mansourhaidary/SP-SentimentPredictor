import streamlit as st 
import matplotlib.pyplot as plt
import re
import nltk
import json
import pickle
import numpy as np
import joblib
import torch
import tensorflow as tf
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from farasa.stemmer import FarasaStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Download NLTK stopwords if not already
nltk.download('stopwords', quiet=True)

########################################
# Initialize Components
########################################

stemmer = FarasaStemmer(interactive=True)
arabert_model_name = "aubmindlab/bert-base-arabertv02"
arabert_prep = ArabertPreprocessor(model_name=arabert_model_name)
arabert_tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_arabert")
arabert_model = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_arabert")
arabert_model.eval()


camelbert_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da"
camelbert_prep = ArabertPreprocessor(model_name=camelbert_model_name)
camelbert_tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\mansour_camelbert")
camelbert_model = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\mansour_camelbert")
camelbert_model.eval()

camelbert_model_namea = "CAMeL-Lab/bert-base-arabic-camelbert-da"
camelbert_prepa = ArabertPreprocessor(model_name=camelbert_model_name)
camelbert_tokenizera = AutoTokenizer.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\camelbert_Abdulrahman")
camelbert_modela = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\manso\\Downloads\\SP2-MODELS\\camelbert_Abdulrahman")
camelbert_modela.eval()



########################################
# Preprocessing Functions
########################################
def initial_preprocess(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[٠١٢٣٤٥٦٧٨٩]', '', text)
    text = text.replace('_', ' ')
    punctuations = r'''÷×؛<>()&^%٪][ـ،/:"؟.,'{}¦+|!”…“–!"#$%&'()+,-./:;<=>?@[\]^{|}'''
    text = re.sub(f'[{re.escape(punctuations)}]', '', text)
    arabic_diacritics = re.compile(r"[ًٌٍَُِّْـ]", re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'گ', 'ك', text)
    text = re.sub(r'س[يى]+ء', 'سيئ', text)
    text = text.replace('♡', '').replace('ﷺ', '')
    text = re.sub(r'\s+', ' ', text).strip()

    def remove_elongation(word):
        word = re.sub(r'(م)\1{2,}', r'\1\1', word)
        word = re.sub(r'([^\sم])\1{1,}', r'\1', word)
        return word

    text = ' '.join([remove_elongation(w) for w in text.split()])
    negation_words = ['لا', 'لن', 'ما', 'ليس', 'بدون', 'غير']
    arabic_stopwords_list = stopwords.words('arabic')
    arabic_stopwords_list = [w for w in arabic_stopwords_list if w not in negation_words]
    words = text.split()
    words = [word for word in words if word not in arabic_stopwords_list]
    text = ' '.join(words)
    return text

def apply_stemming(text):
    return stemmer.stem(text)

def normalize_text_after_stemming(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'گ', 'ك', text)
    text = re.sub(r'س[يى]+ء', 'سيئ', text)
    return text

def full_preprocess(text):
    text = initial_preprocess(text)
    text = apply_stemming(text)
    text = normalize_text_after_stemming(text)
    return text

def preprocess_bilstm(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[٠١٢٣٤٥٦٧٨٩]', '', text)
    text = text.replace('_', ' ')
    punctuations = r'''÷×؛<>()&^%٪][ـ،/:"؟.,'{}¦+|!”…“–!"#$%&'()+,-./:;<=>?@[\]^{|}'''
    text = re.sub(f'[{re.escape(punctuations)}]', '', text)
    arabic_diacritics = re.compile(r"[ًٌٍَُِّْـ]", re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'گ', 'ك', text)
    text = re.sub(r'س[يى]+ء', 'سيئ', text)
    text = text.replace('♡', '').replace('ﷺ', '')
    text = re.sub(r'\s+', ' ', text).strip()

    def remove_elongation(word):
        word = re.sub(r'(م)\1{2,}', r'\1\1', word)
        word = re.sub(r'([^\sم])\1{1,}', r'\1', word)
        return word

    text = ' '.join([remove_elongation(word) for word in text.split()])
    return text

########################################
# Model Predictors
########################################
dummy_label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
def dummy_predict(sentences):
    import random
    return [random.choice(list(dummy_label_map.values())) for _ in sentences]

xgb_label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
def predict_xgb(sentences, model, vectorizer, selector):
    preprocessed = [full_preprocess(s) for s in sentences]
    X_vec = vectorizer.transform(preprocessed)
    X_sel = selector.transform(X_vec)
    preds = model.predict(X_sel)
    return [xgb_label_map[p] for p in preds]



sultan_nb_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
def predict_sultan_nb(sentences, model, vectorizer, selector):
    preprocessed = [full_preprocess(s) for s in sentences]
    X_vec = vectorizer.transform(preprocessed)
    X_sel = selector.transform(X_vec)
    preds = model.predict(X_sel)
    return [sultan_nb_map[p] for p in preds]


arabert_label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
def predict_arabert(sentences):
    preprocessed = [arabert_prep.preprocess(s) for s in sentences]
    inputs = arabert_tokenizer(preprocessed, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = arabert_model(**inputs)
    preds = outputs.logits.argmax(dim=-1).numpy()
    return [arabert_label_map[p] for p in preds]

bilstm_label_map = {0:"Negative", 1:"Positive", 2:"Neutral"}
def predict_bilstm(sentences, model, tokenizer, max_length):
    preprocessed = [preprocess_bilstm(s) for s in sentences]
    seq = tokenizer.texts_to_sequences(preprocessed)
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    probs = model.predict(padded_seq)
    pred_id = probs.argmax(axis=-1)
    return [bilstm_label_map[p] for p in pred_id]




lr_abdul_map = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}

def predict_lr_abdul(sentences, model):
    # Preprocess sentences
    preprocessed = [full_preprocess(s) for s in sentences]
    # Predict using the loaded SVM pipeline (includes vectorizer)
    preds = model.predict(preprocessed)
    # Map to textual labels if needed, assuming preds are strings like "negative", etc.
    return [lr_abdul_map[p] for p in preds]


svm_label_map = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}

def predict_svm(sentences, model):
    # Preprocess sentences
    preprocessed = [full_preprocess(s) for s in sentences]
    # Predict using the loaded SVM pipeline (includes vectorizer)
    preds = model.predict(preprocessed)
    # Map to textual labels if needed, assuming preds are strings like "negative", etc.
    return [svm_label_map[p] for p in preds]




svm_label_map_abd = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}

def predict_svm_abd(sentences, model):
    # Preprocess sentences
    preprocessed = [full_preprocess(s) for s in sentences]
    # Predict using the loaded SVM pipeline (includes vectorizer)
    preds = model.predict(preprocessed)
    # Map to textual labels if needed, assuming preds are strings like "negative", etc.
    return [svm_label_map_abd[p] for p in preds]



bilstm_googleplay_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
def predict_bilstm_googleplay(sentences, model, tokenizer, max_length):
    preprocessed = [preprocess_bilstm(s) for s in sentences]
    # Use texts_to_sequences if tokenizer is a Keras tokenizer
    seq = tokenizer.texts_to_sequences(preprocessed)
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    probs = model.predict(padded_seq)
    pred_id = probs.argmax(axis=-1)
    return [bilstm_googleplay_label_map[p] for p in pred_id]


sultan_rnn_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sultan_rnn(sentences, model, tokenizer, max_length):

    # Preprocess sentences
    preprocessed = [preprocess_bilstm(s) for s in sentences]
    
    # Tokenize sentences
    seq = tokenizer.texts_to_sequences(preprocessed)
    
    # Handle empty sequences
    empty_indices = [i for i, s in enumerate(seq) if len(s) == 0]
    if empty_indices:
        print(f"Warning: Empty sequences found at indices: {empty_indices}")
        
        # Option 1: Replace empty sequences with a placeholder (e.g., <UNK>)
        seq = [
            s if len(s) > 0 else [tokenizer.word_index.get('<UNK>', 0)]  # Replace with <UNK> index or 0
            for s in seq
        ]
        
    # Pad sequences
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    # Ensure padded_seq is a numpy array
    padded_seq = np.array(padded_seq)
    
    # Debugging: Check padded_seq shape
    print(f"Shape of padded_seq: {padded_seq.shape}")
    
    # Predict probabilities
    probs = model.predict(padded_seq)
    
    # Get predicted class IDs
    pred_id = probs.argmax(axis=-1)
    
    # Map predicted IDs to sentiment labels
    return [sultan_rnn_map[p] for p in pred_id]


camelbert_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
def predict_camelbert_googleplay(sentences):
    preprocessed = [camelbert_prep.preprocess(s) for s in sentences]
    inputs = camelbert_tokenizer(preprocessed, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = camelbert_model(**inputs)
    preds = outputs.logits.argmax(dim=-1).numpy()
    return [camelbert_label_map[p] for p in preds]


camelbert_label_mapa = {0: "Negative", 1: "Neutral", 2: "Positive"}
def predict_camelberta(sentences):
    preprocessed = [camelbert_prep.preprocess(s) for s in sentences]
    inputs = camelbert_tokenizera(preprocessed, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = camelbert_modela(**inputs)
    preds = outputs.logits.argmax(dim=-1).numpy()
    return [camelbert_label_mapa[p] for p in preds]




########################################
# Page Config and CSS
########################################
st.set_page_config(page_title="Arabic Sentiment Analysis", layout="centered")

st.markdown("""
<style>
body {
    background-color: black;
    color: white;
}
.sidebar .sidebar-content {
    background: black;
    color: white;
}
.stApp {
    background: black;
    color: white;
}
.centered {
    display: flex;
    justify-content: center;
    margin-bottom: 0px;
}
</style>
""", unsafe_allow_html=True)

# Initialize page state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Main Page'


# Sidebar
st.sidebar.markdown("## About this App")
st.sidebar.write("This app is a sentiment predictor where a user inputs multiple Arabic Sentences to be classified. The models available on this website were the best trained models from the set of models constructed by the Basera team. Each model is unique due to the benchmark dataset that was used to train the model. The benchamrk datasets were obtained from well-known websites: Noon, Booking, TripAdvisor and Googleplay. Lastly, these datasets were enhanced (relabeled) to achieve a higher accuracy in classification ")
st.sidebar.markdown("-  XGBoost: Traditional ML approach\n- **AraBERT: Transformer-based model\n- **BiLSTM: Deep Learning model\n- **SVM: Traditional ML\n- **Logistic Regression: Traditional ML\n- **Naive Bayes: Traditional ML\n- **LSTM: Deep Learning model\n- **CAMeLBERT: Transformer-based model")

# Center the logo
st.markdown('<div class="centered">', unsafe_allow_html=True)
st.image("C:\\Users\\manso\\Downloads\\SP2-MODELS\\b3.png", width=400)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Arabic Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("---")



def show_dataset_page(dataset_name, models_info):
    st.markdown(f"<h2 style='text-align:center; color:white;'>{dataset_name} Dataset</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:white;'>write an introduction</h4>", unsafe_allow_html=True)

    for model_name, accuracy, f1_score in models_info:
        st.markdown(f"<h3 style='text-align:center; color:white;'>{model_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:white;'>Accuracy: {accuracy}, F1-score: {f1_score}</p>", unsafe_allow_html=True)

        # Confusion Matrix placeholder
        fig_cm, ax_cm = plt.subplots()
        fig_cm.patch.set_facecolor('black')
        ax_cm.set_facecolor('black')
        ax_cm.text(0.5, 0.5, "Confusion Matrix Placeholder", ha='center', va='center', color='white', fontsize=12)
        ax_cm.axis('off')
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.pyplot(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

        # ROC Curve placeholder
        fig_roc, ax_roc = plt.subplots()
        fig_roc.patch.set_facecolor('black')
        ax_roc.set_facecolor('black')
        ax_roc.text(0.5, 0.5, "ROC Curve Placeholder", ha='center', va='center', color='white', fontsize=12)
        ax_roc.axis('off')
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.pyplot(fig_roc)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("---")

if st.session_state['page'] == "Noon":
    # Noon.com dataset: (AraBERT, BiLSTM, XGBoost)
    noon_models = [
        ("AraBERT", "00%", "00%"),
        ("BiLSTM", "00%", "00%"),
        ("XGBoost", "00%", "00%")
    ]
    show_dataset_page("Noon.com", noon_models)

elif st.session_state['page'] == "Google":
    # Google Play: (SVM, BiLSTM, AraBERT)
    google_models = [
        ("SVM", "00%", "00%"),
        ("BiLSTM", "00%", "00%"),
        ("CamelBERT", "00%", "00%")
    ]
    show_dataset_page("Google Play", google_models)

elif st.session_state['page'] == "TripAdvisor":
    # TripAdvisor: (Logistic Regression, BiLSTM, CAMeLBERT)
    tripadvisor_models = [
        ("Logistic Regression", "00%", "00%"),
        ("BiLSTM", "00%", "00%"),
        ("CAMeLBERT", "00%", "00%")
    ]
    show_dataset_page("TripAdvisor", tripadvisor_models)

elif st.session_state['page'] == "Booking":
    # Booking.com: (Naive Bayes, LSTM, CAMeLBERT)
    booking_models = [
        ("Naive Bayes", "00%", "00%"),
        ("LSTM", "00%", "00%"),
        ("ARABERT", "00%", "00%")
    ]
    show_dataset_page("Booking.com", booking_models)

else:
    # Main Page
    input_text = st.text_area("Enter multiple Arabic sentences (one per line):", height=200)
    model_choice = st.selectbox(
    "Choose a model:",
    ["XGBoost Noon", "AraBERT Noon", "BiLSTM Noon", "SVM GoooglePlay", "BiLSTM GooglePlay", "camelbert googleplay", "NB Booking", "RNN Booking", "LR Tripadvisor","camelbert Tripadvisor"]
)
    classify_button = st.button("Classify")

    if classify_button:
        sentences = [line.strip() for line in input_text.split('\n') if line.strip()]
        if len(sentences) == 0:
            st.warning("⚠️ Please enter at least one sentence.")
        else:
            if model_choice == "BiLSTM GooglePlay":
                bilstm_googleplay_model_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\mansour_bilstm_model.h5"
                bilstm_googleplay_tokenizer_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\mansour_bilsmt_tokenizer.pkl"
                MAX_LEN = 32
                try:
                     bilstm_googleplay_model = tf.keras.models.load_model(bilstm_googleplay_model_path, compile=False)
                     with open(bilstm_googleplay_tokenizer_path, 'rb') as f:
                        bilstm_googleplay_tokenizer = pickle.load(f)
                except Exception as e:
                    st.error(f"Error loading BiLSTM Google Play model or tokenizer: {e}")
                    st.stop()
                preds = predict_bilstm_googleplay(sentences, bilstm_googleplay_model, bilstm_googleplay_tokenizer, MAX_LEN)
            elif model_choice == "RNN Booking":
                RNN_Booking_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\sultan_rnn.h5"
                RNN_Booking_tokenizer_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\rnn_sultan_tokenizer.pkl"
                MAX_LEN=21
                try:
                    rnn_booking_model = tf.keras.models.load_model(RNN_Booking_path, compile=False)
                    with open(RNN_Booking_tokenizer_path,'rb') as f:
                        rnn_booking_tokenizer=pickle.load(f)
                        preds = predict_sultan_rnn(sentences, rnn_booking_model, rnn_booking_tokenizer, MAX_LEN)
                except Exception as e:
                    st.error(f"Error Loading RNN Booking model {e}")
                    st.stop()

            elif model_choice == "XGBoost Noon":
                xgb_model_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_xgboost.pkl"
                tfidf_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_xgboost_vectorizer.pkl"
                feature_selector_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_feauture_xgboost.pkl"
                try:
                    xgb_model = joblib.load(xgb_model_path)
                    tfidf_vectorizer = joblib.load(tfidf_path)
                    feature_selector = joblib.load(feature_selector_path)
                    preds = predict_xgb(sentences, xgb_model, tfidf_vectorizer, feature_selector)
                except Exception as e:
                    st.error(f"Error loading XGBoost model or components: {e}")
                    st.stop()
            elif model_choice == "AraBERT Noon":
                preds = predict_arabert(sentences)
            elif model_choice=="camelbert googleplay":
                preds = predict_camelbert_googleplay(sentences)
            elif model_choice=="camelbert Tripadvisor":
                preds = predict_camelberta(sentences)
            elif model_choice == "BiLSTM Noon":
                bilstm_model_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\kahlid_bilstm.h5"  # Changed to .h5
                bilstm_tokenizer_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_bilstm_tokenizer.pkl"
                MAX_LENN = 16
                try:
                     bilstm_model = tf.keras.models.load_model(bilstm_model_path, compile=False)
                     with open(bilstm_tokenizer_path, 'rb') as f:
                        bilstm_tokenizer_data = joblib.load(f)        
                except Exception as e:
                    st.error(f"Error loading BiLSTM model or tokenizer: {e}")
                    st.stop()
                preds = predict_bilstm(sentences, bilstm_model, bilstm_tokenizer_data, MAX_LENN)
            elif model_choice=="NB Booking":
                    NB_Booking_path_model = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_xgboost.pkl"
                    tfidf_path_NB_booking = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\khalid_xgboost_vectorizer.pkl"
                    feature_selector_path_booking_nb = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\SP2-MODELS\\khalid_feauture_xgboost.pkl"
                    try:
                        NB_Booking_model = joblib.load(NB_Booking_path_model)
                        tfidf_NB_booking = joblib.load(tfidf_path_NB_booking)
                        nb_feature=joblib.load(feature_selector_path_booking_nb)
                    except Exception as e:
                        st.error(f"Error loading XGBoost model or components: {e}")
                        st.stop()
                    preds = predict_sultan_nb(sentences,NB_Booking_model,tfidf_NB_booking,nb_feature)    
            elif model_choice=="SVM GoooglePlay":
                # SVM Model
                svm_model_path = "C:\\Users\\manso\\Downloads\\SP2-MODELS\\\Mansour_svm_model.pkl"
                try:
                    svm_model = joblib.load(svm_model_path)
                except Exception as e:
                    st.error(f"Error loading SVM model: {e}")
                    st.stop()
                preds = predict_svm(sentences, svm_model)
            elif model_choice=="LR Tripadvisor":
                lr_path="CC:\\Users\\manso\\Downloads\\SP2-MODELS\\Mansour_svm_model.pkl"
                try:
                    lr_model=joblib.load(lr_path)
                except Exception as e:
                    st.error(f"error laoding")
                    st.stop
                preds = predict_lr_abdul(sentences,lr_model)        
                
            positive_sentences = [s for s, p in zip(sentences, preds) if p == "Positive"]
            negative_sentences = [s for s, p in zip(sentences, preds) if p == "Negative"]
            neutral_sentences = [s for s, p in zip(sentences, preds) if p == "Neutral"]

            pos_count = len(positive_sentences)
            neg_count = len(negative_sentences)
            neu_count = len(neutral_sentences)

            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [pos_count, neg_count, neu_count]

            if sum(sizes) == 0:
                sizes = [1,1,1]

            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            explode = (0.05, 0.05, 0.05)
            colors = ['green', 'red', 'yellow']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   explode=explode, colors=colors, textprops={'color':'white'})
            ax.axis('equal')

            centre_circle = plt.Circle((0,0),0.70,fc='black')
            fig.gca().add_artist(centre_circle)

            st.markdown("<h3 style='text-align: center; color: white;'>Classification Results</h3>", unsafe_allow_html=True)
            st.pyplot(fig)

            with st.expander("Positive Sentences"):
                if pos_count > 0:
                    for ps in positive_sentences:
                        st.write(f"• {ps}")
                else:
                    st.write("No positive sentences.")

            with st.expander("Negative Sentences"):
                if neg_count > 0:
                    for ns in negative_sentences:
                        st.write(f"• {ns}")
                else:
                    st.write("No negative sentences.")

            with st.expander("Neutral Sentences"):
                if neu_count > 0:
                    for ns in neutral_sentences:
                        st.write(f"• {ns}")
                else:
                    st.write("No neutral sentences.")