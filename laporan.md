# Laporan Proyek Machine Learning - Cindy Maharani
## Project Overview
Sitem rekomendasi saat ini sedang menjadi tren. Saat ini berbelanja online merupakan pilihan yang lebih disukai masyarakat karena kemudahannya dan kecepatannya (Mathew, Praveena  et al., 2016). Pilihan barang - barang yang tersedia online juga semakin banyak, salah satunya adalah buku. Akan tetapi, pilihan yang beragam terkadang membuat konsumen kesulitan untuk memilik buku sesuai dengan preferensi mereka, baik buku secara fisik maupun buku elektronik (e-book). Mereka dapat membaca sinopisis yang tersedia atau sample halaman yang ada pada buku elektronik, tetapi hal tersebut akan memakan banyak waktu jika terdapat banyak buku yang dipertimbangkan. Oleh karena itu, dibutuhkan sistem rekomendasi buku yang membantu pengguna dalam memilih buku yang sesuai dengan preferensi mereka.

Selain untuk membantu pengguna, hal ini juga memberikan keuntungan untuk situs jual beli online yang menjual buku baik. Dengan merekomendasikan buku - buku yang kemungkinan akan dibeli oleh pengguna maka pendapatan mereka akan bertambah.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku berbasis data menggunakan dua pendekatan populer, yaitu content-based filtering dan collaborative filtering, dengan memanfaatkan dataset Book-Crossing.

Sumber: P. Mathew, B. Kuriakose and V. Hegde, "Book Recommendation System through content based and collaborative filtering method," 2016 International Conference on Data Mining and Advanced Computing (SAPIENCE), Ernakulam, India, 2016, pp. 47-52, doi: 10.1109/SAPIENCE.2016.7684166.

## Business Understanding

### Problem Statements
- Bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating buku, bagaimana perusahaan dapat merekomendasikan buku - buku lainnya yang mungkin disukai dan belum pernah dibaca oleh pengguna? 

### Goals
- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dibaca/ di-_rating_ sebelumnya dengan teknik collaborative filtering.

### Solution statements
Untuk mencapai tujuan tersebut, digunakan dua pendekatan:
- Content-Based Filtering: Rekomendasi didasarkan pada kesamaan fitur buku (judul, penulis, penerbit.).
- Collaborative Filtering: Rekomendasi didasarkan pada interaksi pengguna dan pola penilaian mereka terhadap buku.

## Data Understanding
Dataset ini dikumpulkan oleh Cai-Nicolas Ziegler dalam penelusuran selama 4 minggu (Agustus/September 2004) dari komunitas Book-Crossing dengan izin dari Ron Hornbaker, CTO Humankind Systems. Berisi 278.858 pengguna (anonim tetapi dengan informasi demografis) yang memberikan 1.149.780 penilaian (eksplisit/implisit) untuk sekitar 271.379 buku.
Dataset yang digunakan adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Datset ini terdiri dari tiga file, berikut keterangan variabel - variabel pada setiap file
Variabel-variabel pada Books.csv dataset adalah sebagai berikut:
- ISBN: Kode identifikasi unik untuk setiap buku.
- Book-Title: Judul Buku.
- Book-Author: Nama penulis buku. Jika sebuah buku memiliki beberapa penulis, dataset ini hanya mencantumkan penulis pertama.
- Year-Of-Publication: Tahun buku diterbitkan.
- Publisher: Nama penerbit buku.
- Image-URL-S: URL (alamat web) yang mengarah ke gambar sampul buku berukuran kecil
- Image-URL-M: URL (alamat web) yang mengarah ke gambar sampul buku berukuran sedang
- Image-URL-L: URL (alamat web) yang mengarah ke gambar sampul buku berukuran besar

Variabel-variabel pada Users.csv dataset adalah sebagai berikut:
- User-ID: Kode unik mengidentifikasi setiap pengguna dalam dataset.
- Location: Lokasi geografis pengguna.
- Age: Usia pengguna. 

Variabel-variabel pada Ratings.csv dataset adalah sebagai berikut:
- User-ID: Kode unik mengidentifikasi setiap pengguna dalam dataset, yang merujuk pada id pegguna di tabel Users.csv
- ISBN: Kode identifikasi unik untuk setiap buku yang terdapat pada Books.csv
- Book-rating: Rating pengguna dari 0 - 10. (0 adalah rating implisit saat pengguna berinteraksi dengan buku, 1-10 adalah rating eksplisit yang langsung diberikan oleh pengguna)

### Univariate Exploratory Data Analysis
1. Books.csv
    - Dataset ini berisi 271.360 baris dan 8 kolom
    - Dataset ini memiliki 4 nilai null (2 pada kolom Book-Author dan 2 pada Publisher)
    - Data pada dataset ini ada 3 baris yang memiliki missfielded value karena data Book-Author yang tergabung pada data Book-Title.
    - Selain itu, data tahun publikasi juga memiliki dua tipe data berbeda, string dan integer.
2. Users.csv
    - Dataset ini memiliki 278.858 baris dan 3 kolom
    - Data Age pada dataset ini banyak memiliki nilai kosong sehingga diperlukan imputasi.
    - Selain itu, data Age juga memiliki nilai yang tidak masuk akal untuk seorang pengguna yang melakukan review buku, yaitu di bawah 5 tahun dan diatas 90 tahun.
      </br><img width="642" alt="Screenshot 2025-05-01 at 23 42 19" src="https://github.com/user-attachments/assets/309a21d8-c260-4a3a-acab-e720b4315a20" />

3. Ratings.csv
    - Dataset ini memiliki 1.149.780 baris dan 3 kolom
    - Data Book-Rating memiliki dua jenis rating, yaitu rating eksplisit (penilaian langsung oleh pengguna dalam bentuk angka pada skala 1 hingga 10) dan rating implisit (penilaian yang tidak diberikan secara langsung, tetapi mengindikasikan bahwa pengguna mungkin telah berinteraksi dengan buku tersebut). Pada sistem rekomendasi yang akan dibangun berdasarkan collaborative filtering, data rating yang digunakan hanyalah rating eksplisit, sehingga rating implisit akan didrop.

## Data Preparation
1. Menghapus kolom Image-URL-S, Image-URL-S, dan Image-URL-S karena pendekatan content based filtering yang akan dibangun berbasis teks, tidak melihat kemiripan cover buku. Menghapus kolom menggunakan fungsi drop().
2. Imputasi nilai null pada kolom Book-Author dan Publisher dengan nilai 'other' untuk memastikan bahwa semua data memiliki nilai, sehingga dapat diproses oleh algoritma.
3. Memperbaiki nilai yang bergeser (missfielded value) dengan mengisi nilai yang sesuai secara manual. Hal ini penting karena terkait dengan integritas data, analisis yang akurat, dan keberhasilan model machine learning.
4. Menyamakan tipe data tahun publikasi dan memperbaiki nilai yang tidak masuk akal (tahun diatas 2025) dengan data tahun median.
5. Memperbaiki nilai umur yang kosog dan umur tidak masuk akal, umur di bawah 5 tahun dan diatas 90 tahun, dengan cara assign nilainya menjadi NaN, lalu imputasi semua nilai NaN dengan nilai random dalam rentang 20-60. Berikut distribusi umur setelah nilai NaN dan nilai ekstrim diganti dengan nilai random: </br><img width="654" alt="Screenshot 2025-05-01 at 23 58 18" src="https://github.com/user-attachments/assets/5f54dd27-24e0-4b90-a247-f1e66962ccaf" />
6. Drop data rating implisit dan hanya gunakan data rating eksplisit untuk keperluan membuat sistem rekomendasi.
7. Menggabungkan books dan ratings karena memuat kolom - kolom penting untuk membuat sistem rekomendasi. Kolom - kolom tersebut adalah: User-ID, ISBN, Book-Rating,	Book-Title, Book-Author, Year-Of-Publication, dan Publisher. Saya menggabungkan kedua tabel dengan inner join karena saya ingin mengambil baris- baris yang terdapat pada kedua tabel.
8. Berikut ini visualisasi pada data ratings eksplisit yang sudah dibersihkan.</br><img width="724" alt="Screenshot 2025-05-02 at 16 11 51" src="https://github.com/user-attachments/assets/52c768be-af7c-44a4-8415-2d65ec916efd" />
9. Selanjutnya adalah persiapan data untuk pemodelan. Pada bagian ini, saya hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Penghapusan data duplikat menggunakan fungsi drop_duplicates() pada kolom ISBN. Selanjutnya, konversi data series menjadi list menggunakan fungsi tolist() dari library numpy. Lalu, membuat dictionary untuk menentukan pasangan key-value pada data ISBN, Book-Title, Book-Author dan Publisher.
10. Berikut ini visualisasi dari data jumlah buku yang diterbitkan per tahun setelah hasil drop duplicates pada ISBN.</br><img width="1068" alt="Screenshot 2025-05-02 at 16 12 00" src="https://github.com/user-attachments/assets/8139feab-cec5-4376-9fc3-2cf5e1c03db6" />
11. Membuat kolom baru bernama "features" yang berisi gabungan string book-title, book-author, dan publisher untuk keperluan content based filtering.
12. Saya menggunakan TF-IDF Vectorizer pada kolom features untuk menemukan representasi fitur penting dari setiap atribut buku.
13. 

## Modeling
### Content Based Filtering
Algoritma ini merekomendasikan item yang mirip dengan preferensi pengguna di masa lalu. Kelebihan dari algoritma ini adalah tidak memerlukan data pengguna lain karena hanya membuuhkan data - data setiap item untuk dicocokan dengan item yang mirip dengan item yang disukai pengguna di masa lalu. Kelemahan dari algoritma ini adalah rekomendasi yang dihasilkan hanya terbatas pada item yang serupa. Contohnya, ketika pengguna menyukai buku bertema romantis, maka rekomendasi yang dihasilkan sebatas buku - buku yang bertema romantis juga.

Pada kasus ini, saya menggunakan cosine similarity untuk menghitung derajat kesamaan (similarity degree) antar "features" buku dengan teknik cosine similarity. Top-5 buku paling mirip ditampilkan berdasarkan buku yang disukai pengguna. Berikut contoh mencari 5 buku termirip dari buku "The Girl Who Loved Tom Gordon : A Novel" ISBN: 0684867621, dengan penulis Stephen King dan penerbit scribner:
| book_title                  | idbook     | author       | publisher |
|-----------------------------|------------|--------------|-----------|
| The Girl Who Loved Tom Gordon | 0671042858 | Stephen King | Pocket  |
| From a Buick 8 : A Novel    | 0743211375 | Stephen King | Scribner  |
| Dreamcatcher                | 074343627X | Stephen King | Pocket    |
| Dreamcatcher                | 0743211383 | Stephen King | Scribner  |
| Dreamcatcher                | 0743467523 | Stephen King | Pocket    |

### Collaborative Filtering
Untuk menyelesaikan permasalahan skalabilitas dan personalisasi dalam sistem rekomendasi, digunakan pendekatan collaborative filtering berbasis deep learning. Model ini dibangun menggunakan TensorFlow dan Keras, dengan merancang sebuah arsitektur bernama RecommenderNet. 

Model ini bekerja dengan cara mengubah identitas pengguna dan buku menjadi representasi vektor (embedding), lalu menghitung skor prediksi rating melalui operasi dot product antar vektor user dan book, serta penambahan bias masing-masing.

Langkah-langkah penting dalam modeling adalah:
- Encode user dan ISBN menjadi integer.
- Buat dataset pasangan (user, book) sebagai input dan rating sebagai output.
- Bangun model RecommenderNet dengan embedding layer untuk user dan book.
- Gunakan fungsi aktivasi sigmoid agar output berada dalam rentang [0,1].
- Loss function: Binary Crossentropy. Optimizer: Adam. Evaluation metric: Root Mean Squared Error (RMSE).

Setelah model dilatih, dilakukan inference untuk menghasilkan top-10 rekomendasi buku bagi pengguna yang diambil secara acak. Buku yang pernah diberi rating oleh pengguna akan dikecualikan. Rekomendasi diberikan berdasarkan skor prediksi tertinggi dari hasil model. Dengan demikian, sistem ini mampu menyelesaikan permasalahan pencarian buku yang relevan secara efisien dan personal dengan memberikan top-N rekomendasi untuk setiap pengguna.

Kelebihan algoritma ini adalah mampu menangkap pola kompleks dan memberikan rekomendasi yang personal. Kekurangannya adalah, walaupun algoritma ini tidak membutuhkan data - data atau informasi mengenai item yang ada, tetapi algoritma ini membutuhkan data rating komunitas pengguna yang tentunya lebih banyak jumlahnya dan proses pelatihan yang lebih besar.

Berikut contoh hasil top 10 rekomendasi untuk pengguna 243891

Buku dengan Rating Tinggi dari Pengguna
|  Judul  |	Penulis  |	Penerbit |
|---------|----------|-----------|
|Tears of the Giraffe (No.1 Ladies Detective Agency)| Alexander McCall Smith|	Anchor|
|The Secret Life of Bees | Sue Monk Kidd |	Penguin Books|
|Empire Falls |	Richard Russo |	Vintage Books USA|
|Timeline	| MICHAEL CRICHTON | Ballantine Books|

Top 10 Rekomendasi Buku
| No. | Judul                                                  | Penulis          | Penerbit            |
| :-- | :----------------------------------------------------- | :--------------- | :------------------ |
| 1   | Charlotte's Web (Trophy Newbery)                       | E. B. White      | HarperTrophy        |
| 2   | Harry Potter and the Prisoner of Azkaban (Book 3)      | J. K. Rowling    | Scholastic          |
| 3   | Harry Potter and the Goblet of Fire (Book 4)           | J. K. Rowling    | Scholastic Paperbacks |
| 4   | Harry Potter and the Prisoner of Azkaban (Book 3)      | J. K. Rowling    | Scholastic          |
| 5   | Harry Potter and the Goblet of Fire (Book 4)           | J. K. Rowling    | Scholastic          |
| 6   | Harry Potter and the Sorcerer's Stone (Book 1)         | J. K. Rowling    | Scholastic          |
| 7   | The Two Towers (The Lord of the Rings, Part 2)         | J.R.R. TOLKIEN   | Del Rey             |
| 8   | Dune (Remembering Tomorrow)                            | Frank Herbert    | ACE Charter         |
| 9   | Harry Potter and the Sorcerer's Stone (Book 1)         | J. K. Rowling    | Scholastic          |
| 10  | The Return of the King (The Lord of the Rings, Part 3) | J.R.R. TOLKIEN   | Del Rey             |


## Evaluation
### Metrik Evaluasi
Metrik Evaluasi:
Untuk mengukur performa dari kedua pendekatan sistem rekomendasi, digunakan beberapa metrik evaluasi:
**1. RMSE (Root Mean Squared Error)**
Digunakan untuk mengevaluasi performa collaborative filtering dalam memprediksi rating:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Dimana:
- $RMSE$ adalah nilai Root Mean Squared Error. Semakin rendah nilai RMSE, maka semakin baik model dalam memprediksi nilai rating.
- $n$ adalah jumlah total data point atau observasi (misalnya, jumlah rating yang dievaluasi).
- $y_i$ adalah nilai aktual untuk data point ke-$i$ (contohnya, rating sebenarnya yang diberikan oleh pengguna).
- $\hat{y}_i$ adalah nilai yang diprediksi oleh model atau sistem untuk data point ke-$i$ (contohnya, rating yang diprediksi oleh sistem rekomendasi).
- $\sum_{i=1}^{n}$ adalah simbol sigma yang menandakan penjumlahan dari semua selisih kuadrat, dimulai dari $i=1$ hingga $n$.

2. Precision@K dan Recall@K (Collaborative Filtering)
Digunakan untuk mengukur relevansi dari hasil rekomendasi:
- Precision@K: Persentase dari Top-K item yang direkomendasikan dan benar-benar relevan bagi pengguna.
- Recall@K: Persentase dari seluruh item relevan yang berhasil direkomendasikan dalam Top-K hasil.

3. Precision (Content-Based Filtering)
Digunakan untuk mengevaluasi relevansi hasil rekomendasi berbasis konten:

$$
P = \frac{\text{ of our recommendations that are relevant}}{\text{ of items we recommended}}
$$

Dimana:
- $P$ adalah nilai Precision.
- *# of our recommendations that are relevant* adalah jumlah rekomendasi yang diberikan oleh sistem yang relevan bagi pengguna.
- *# of items we recommended* adalah jumlah total item yang direkomendasikan oleh sistem.

Metrik ini sesuai dengan konteks di mana tidak tersedia rating eksplisit dari pengguna. Rekomendasi dinilai relevan jika kontennya (judul, penulis, penerbit) sesuai dengan item acuan.

### Hasil Evaluasi Collaborative Filtering:

- RMSE (Validation): 0.1991 — menunjukkan prediksi rating cukup akurat.
- Precision@10: 0.8389 — 84% dari top-10 rekomendasi sesuai preferensi pengguna.
- Recall@10: 0.6089 — sistem berhasil merekomendasikan lebih dari 60% buku yang relevan.

### Hasil Evaluasi Content-Based Filtering:

Contoh kasus pada buku "The Girl Who Loved Tom Gordon : A Novel" ISBN: 0684867621, dengan penulis Stephen King dan penerbit scribner:

| book_title                  | idbook     | author       | publisher |
|-----------------------------|------------|--------------|-----------|
| The Girl Who Loved Tom Gordon | 0671042858 | Stephen King | Pocket  |
| From a Buick 8 : A Novel    | 0743211375 | Stephen King | Scribner  |
| Dreamcatcher                | 074343627X | Stephen King | Pocket    |
| Dreamcatcher                | 0743211383 | Stephen King | Scribner  |
| Dreamcatcher                | 0743467523 | Stephen King | Pocket    |

Dari 5 rekomendasi yang diberikan, 5 di antaranya adalah buku karya Stephen King dan diterbitkan oleh penerbit yang sama atau mirip, menunjukkan konsistensi konten.

- Total rekomendasi = 5
- Rekomendasi yang relevan = 5
- Precision = 5 / 5 = 1.0

Ini menunjukkan bahwa 100% rekomendasi yang diberikan sistem berbasis konten relevan, mengindikasikan bahwa model efektif dalam mengenali kesamaan konten.

## 1. Apakah Hasil Model Menjawab Setiap *Problem Statement*?

### ✅ Problem Statement 1:
**Bagaimana cara membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?**

**Jawaban:**  
Ya. Model content-based filtering yang dibangun berhasil menghasilkan rekomendasi yang dipersonalisasi berdasarkan fitur buku. Dalam contoh kasus buku "The Girl Who Loved Tom Gordon", sistem merekomendasikan 5 buku lain yang semuanya ditulis oleh penulis yang sama (Stephen King) dan diterbitkan oleh penerbit serupa. Ini menunjukkan kemampuan sistem dalam mengidentifikasi kesamaan konten untuk personalisasi, dengan precision mencapai 1.0 (100% relevan).

---

### ✅ Problem Statement 2:
**Dengan data rating buku, bagaimana perusahaan dapat merekomendasikan buku-buku lainnya yang mungkin disukai dan belum pernah dibaca oleh pengguna?**

**Jawaban:**  
Ya. Model collaborative filtering yang digunakan mampu merekomendasikan buku berdasarkan pola rating pengguna lain yang serupa. Hasil evaluasi menunjukkan Precision@10 sebesar 0.8389 (sekitar 84% dari 10 rekomendasi teratas relevan) dan Recall@10 sebesar 0.6089 (lebih dari 60% item relevan berhasil direkomendasikan). Nilai RMSE validasi yang rendah (0.1991) juga mengindikasikan akurasi prediksi rating yang baik. Ini membuktikan model efektif dalam menyarankan buku baru yang kemungkinan besar disukai pengguna berdasarkan perilaku kolektif.

---

## 2. Apakah Berhasil Mencapai Setiap *Goal* yang Diharapkan?

### ✅ Goal 1:
**Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.**

**Status:**  
Tercapai. Sistem content-based filtering telah berhasil dibuat dan mampu memberikan rekomendasi buku berdasarkan kesamaan fitur (konten) buku, seperti yang ditunjukkan pada contoh evaluasi dengan precision 1.0.

---

### ✅ Goal 2:
**Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dibaca/ di-rating sebelumnya dengan teknik collaborative filtering.**

**Status:**  
Tercapai. Sistem collaborative filtering berhasil menghasilkan rekomendasi buku yang relevan (Precision@10 = 0.8389, Recall@10 = 0.6089) berdasarkan preferensi historis pengguna dan pengguna lain, serta mampu memprediksi rating dengan cukup akurat (RMSE = 0.1991).

---

## 3. Apakah Setiap *Solution Statement* Berdampak dan Efektif?

### 🔹 Solution 1:
**Menggunakan Content-Based Filtering: Rekomendasi didasarkan pada kesamaan fitur buku (judul, penulis, penerbit.).**

**Evaluasi Dampak:**  
- Pendekatan ini terbukti efektif dalam menemukan item yang secara intrinsik mirip.
- Dalam kasus uji, model berhasil mengidentifikasi buku-buku lain dari penulis yang sama (Stephen King), menunjukkan pemahaman konteks dan fitur yang kuat.
- Tingkat presisi 1.0 pada kasus uji menegaskan bahwa rekomendasi yang dihasilkan sangat relevan berdasarkan konten, cocok untuk pengguna yang mencari buku serupa dengan yang sudah mereka sukai.
- Metode ini independen dari data pengguna lain, berguna saat data interaksi pengguna terbatas.

---

### 🔹 Solution 2:
**Menggunakan Collaborative Filtering: Rekomendasi didasarkan pada interaksi pengguna dan pola penilaian mereka terhadap buku.**

**Evaluasi Dampak:**  
- Pendekatan ini efektif dalam menangkap preferensi pengguna yang mungkin tidak terlihat hanya dari fitur buku (serendipity).
- Metrik evaluasi (Precision@10 = 0.8389, Recall@10 = 0.6089) menunjukkan bahwa sebagian besar rekomendasi relevan dan sistem mampu menjangkau item-item yang disukai pengguna.
- RMSE yang rendah (0.1991) menunjukkan kemampuan prediksi rating yang baik, meningkatkan kepercayaan pada kualitas rekomendasi.
- Metode ini mampu merekomendasikan item yang beragam (tidak harus mirip secara konten) selama disukai oleh pengguna dengan selera serupa.
- Efektivitasnya bergantung pada ketersediaan data interaksi pengguna yang cukup.
---

## Kesimpulan
Hasil evaluasi sistem rekomendasi buku sepenuhnya mendukung tujuan awal proyek:

✅ Semua problem statement terjawab dengan baik melalui implementasi kedua teknik filtering (Content-Based dan Collaborative).
🎯 Kedua goals — menghasilkan rekomendasi personal berbasis konten dan rekomendasi relevan berbasis preferensi pengguna — tercapai secara penuh, didukung oleh metrik evaluasi yang solid (Precision, Recall, RMSE) dan contoh kasus yang menunjukkan relevansi tinggi.
🚀 Solusi yang diterapkan, yaitu penggunaan Content-Based Filtering dan Collaborative Filtering, terbukti efektif dan berdampak positif dalam menyediakan rekomendasi buku yang akurat dan sesuai dengan kebutuhan pengguna.
Kedua pendekatan menunjukkan nilai yang signifikan: Content-Based Filtering terbukti sangat andal dalam menemukan buku-buku dengan kesamaan fitur (seperti presisi 1.0 pada contoh kasus), ideal untuk pengguna yang mencari item serupa. Sementara itu, Collaborative Filtering menunjukkan kinerja kuat dalam menyajikan rekomendasi baru yang relevan berdasarkan pola perilaku pengguna (Precision@10 ≈ 84%, Recall@10 ≈ 61%), efektif untuk discovery.

Kombinasi kedua metode ini sangat potensial untuk diimplementasikan pada platform buku digital atau e-commerce guna meningkatkan pengalaman pengguna, memperkaya penemuan buku, dan mendorong engagement secara keseluruhan.
