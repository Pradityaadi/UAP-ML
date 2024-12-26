import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AutoTokenizer, TFBertForSequenceClassification

# Fungsi untuk prediksi menggunakan model LSTM
def predict_sentiment_lstm(model, tokenizer, comment, max_len=100):
    sequences = tokenizer.texts_to_sequences([comment])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    sentiment_classes = ['Negatif', 'Netral', 'Positif']
    predicted_sentiment = sentiment_classes[np.argmax(prediction)]
    return predicted_sentiment

# Fungsi untuk prediksi menggunakan model BERT
def predict_sentiment_bert(model, tokenizer, comment):
    inputs = tokenizer(comment, return_tensors="tf", max_length=128, truncation=True, padding="max_length")
    outputs = model(inputs)
    prediction = np.argmax(outputs.logits.numpy(), axis=1)
    sentiment_classes = ['Negatif', 'Netral', 'Positif']
    predicted_sentiment = sentiment_classes[prediction[0]]
    return predicted_sentiment

# Halaman utama aplikasi
def main():
    st.title("Analisis Sentimen Komentar")
    st.write("Aplikasi ini menganalisis sentimen komentar Anda (Positif, Netral, atau Negatif) menggunakan dua pilihan model: LSTM dan BERT.")

    # Pilihan model
    model_choice = st.radio("Pilih Model untuk Analisis Sentimen:", ["LSTM", "BERT"])

    # Path untuk model dan tokenizer
    if model_choice == "LSTM":
        tokenizer_path = "model/tokenizer.joblib"
        model_path = "model/model_lstm.h5"
    else:  # BERT
        tokenizer_path = "model/model_bert"
        model_path = "model/model_bert"

    # Cek keberadaan file tokenizer dan model
    try:
        if model_choice == "LSTM":
            tokenizer = joblib.load(tokenizer_path)
            model = load_model(model_path)
        else:  # BERT
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = TFBertForSequenceClassification.from_pretrained(model_path)
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau tokenizer: {e}")
        return

    # Input komentar
    user_input = st.text_area("Masukkan komentar di sini:")

    # Tombol prediksi
    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Harap masukkan komentar terlebih dahulu!")
        else:
            if model_choice == "LSTM":
                predicted_sentiment = predict_sentiment_lstm(model, tokenizer, user_input)
            else:  # BERT
                predicted_sentiment = predict_sentiment_bert(model, tokenizer, user_input)
            st.success(f"Sentimen untuk komentar ini adalah: **{predicted_sentiment}**")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
