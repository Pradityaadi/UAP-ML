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
   git remote add origin https://github.com/Pradityaadi/UAP-ML.git
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
   streamlit run src/app.py
   ```
---

## 🏷️ Deskripsi Model
### LSTM
#### Preprocessing
Proses preprocessing dalam proyek ini dimulai dengan pembacaan data dari file CSV, di mana rating diambil dari tag HTML dalam kolom 'Rating' menggunakan fungsi `extract_rating()` dan diubah menjadi skala 1 hingga 5. Selanjutnya, baris dengan nilai kosong pada kolom 'Review' atau 'Rating' dihapus. Rating kemudian diklasifikasikan menjadi sentimen positif (4-5) dan negatif (1-3) menggunakan fungsi `apply()`. Setelah itu, teks ulasan di-tokenisasi menggunakan Tokenizer dari Keras, mengonversi teks menjadi urutan angka dengan ukuran kosakata maksimum 20.000 kata, dan panjang urutan dipadatkan dengan `pad_sequences()` hingga mencapai 100 token. Data kemudian dibagi menjadi set pelatihan dan pengujian menggunakan `train_test_split()` dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian, siap digunakan untuk model klasifikasi sentimen berbasis LSTM.

#### Evaluation
![image](https://github.com/user-attachments/assets/2348d31d-c5aa-4e80-a448-dc0760d8d3ab)

Gambar di atas merupakan Classification Report dari model setelah dilakukan prediksi terhadap testing set. Dapat dilihat bahwa akurasinya mencapai 96%. Pada label 'Negatif' (0), model mencapai precision 83%, recall 61%, dan f1-score 71%, yang menunjukkan bahwa meskipun model cukup baik dalam mendeteksi kelas ini, ada beberapa kesalahan dalam identifikasi prediksi. Sementara itu, pada label 'Positif' (1), model berhasil mencapai precision 97%, recall 99%, dan f1-score 98%, menunjukkan kinerja yang sangat baik dalam mengklasifikasikan kelas ini. Secara keseluruhan, model menunjukkan performa yang sangat baik dengan f1-score rata-rata 0.96 dan akurasi 96%, yang mengindikasikan kemampuannya dalam menangani kedua kelas dengan baik, terutama untuk kelas 'Positif'.

### BERT
#### Preprocessing
Pada proyek ini, preprocessing dilakukan untuk mempersiapkan data teks ulasan hotel agar dapat digunakan dalam model BERT. Data teks pertama-tama diubah menjadi format yang dapat diterima oleh model BERT dengan menggunakan tokenizer dari Hugging Face. Tokenizer ini mengonversi teks menjadi ID token yang sesuai dengan kosakata model BERT, serta menambahkan attention mask untuk mengidentifikasi token yang relevan. Data kemudian dibagi menjadi set pelatihan dan validasi menggunakan `train_loader` dan `val_loader`, yang merupakan DataLoader dari PyTorch yang memungkinkan pemrosesan batch secara efisien. Proses ini mencakup padding atau pemotongan panjang input agar konsisten, serta encoding label ke dalam format yang sesuai untuk tugas klasifikasi. Dengan langkah-langkah ini, data siap digunakan dalam pelatihan model BERT untuk klasifikasi sentimen.

#### Evaluation
![image](https://github.com/user-attachments/assets/c05aed41-8f44-46fa-a5f5-86e3d8eea82a)

Gambar di atas merupakan Classification Report dari model setelah dilakukan prediksi terhadap testing set. Dapat dilihat bahwa akurasinya mencapai 92%. Pada label '0', model berhasil mencapai precision, recall, dan f1-score 100%, menunjukkan kinerja yang sempurna untuk kelas ini. Label '5' juga menunjukkan performa yang sangat baik dengan precision 92%, recall 94%, dan f1-score 93%, yang menunjukkan prediksi yang sangat akurat untuk kelas tersebut. Namun, pada label '2', model memiliki precision yang rendah (21%) dan recall yang hanya 27%, yang menunjukkan kesulitan model dalam mengidentifikasi kelas ini dengan baik. Label '1', '3', dan '4' menunjukkan performa yang lebih baik dengan f1-score masing-masing 0.76, 0.49, dan 0.54, meskipun masih ada ruang untuk perbaikan. Secara keseluruhan, model menunjukkan performa yang baik dengan f1-score rata-rata 0.92, namun dengan beberapa tantangan dalam menangani kelas-kelas yang lebih sedikit atau lebih sulit dikenali.

## 📝 Hasil dan Analisis 

| Model      |     Accuracy      | 
|------------|-------------------|
| LSTM       |       96%         |
| BERT       |       92%         | 

1. Akurasi:

   - LSTM:
     Akurasi pada training dan testing mencapai 96%, yang menunjukkan bahwa model LSTM memberikan hasil yang sangat baik pada kedua set data. Namun, ada sedikit penurunan pada beberapa label yang lebih jarang ditemukan dalam data, meskipun performanya secara keseluruhan sangat solid.

   - BERT:
     Akurasi pada BERT mencapai 92%, sedikit lebih rendah dibandingkan LSTM. Meskipun akurasi testing BERT sangat baik, model ini menunjukkan sedikit penurunan dalam klasifikasi beberapa label dengan jumlah data yang lebih sedikit.



**Kesimpulan:**
Dari tabel perbandingan performa model, dapat dilihat bahwa LSTM menunjukkan akurasi yang lebih tinggi (96%) dibandingkan dengan BERT (92%) pada data training dan testing. LSTM lebih unggul dalam menangani data yang lebih besar dan lebih kompleks, sementara BERT meskipun efektif, memiliki sedikit penurunan performa pada beberapa label. LSTM lebih cocok untuk aplikasi yang membutuhkan akurasi tinggi dan generalisasi yang baik pada berbagai data, sementara BERT dapat dipertimbangkan jika model yang lebih ringan dan efisien dibutuhkan, meskipun dengan akurasi yang sedikit lebih rendah.
