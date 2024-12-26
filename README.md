# Analisis Sentimen Hotel Reviews from TripAdvisor
## 🗂️ Deskripsi Projeck

Proyek ini bertujuan untuk menganalisis sentimen dari ulasan hotel yang ada di TripAdvisor menggunakan teknik pemrosesan bahasa alami (NLP). Dengan memanfaatkan dataset ulasan hotel, proyek ini akan mengklasifikasikan sentimen ulasan menjadi positif, negatif, atau netral. Analisis ini dapat membantu pihak hotel memahami umpan balik pelanggan, mengidentifikasi area yang perlu perbaikan, serta meningkatkan pengalaman pelanggan. Metode yang digunakan meliputi analisis teks, pembersihan data, dan penerapan model pembelajaran mesin untuk mengidentifikasi sentimen yang terkandung dalam ulasan hotel.

Projeck ini menggunakan Dataset yang diambil dari [Kaggle](https://www.kaggle.com/datasets/ruchibhadauria/hotel-reviews-from-tripadvisor)


## 🖇️ Langkah Instalasi
1. **Clone Repository:**
   ```bash
   git init
   git add .
   git commit -m "Inisialisasi proyek"
   git remote add origin 
   git branch -M main
   git push -u origin main

   commit
   git status
   git add (sesuai file yang ditambahkan)
   git commit -m "coba"
   git push origin main
  
   ```

2. **Buat Virtual Environment:**
   ```bash
   python -m venv env
   env\Scripts\activate   # Untuk Windows
   ```

3. **Instal Dependencies:**
   ```bash
   pip install pdm
   pdm init
   pdm add streamlit
   pdm add tensorflow
   pdm add joblib
   pdm add scikit-learn
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Web:**
   ```bash
   pdm run start
   ```
---
