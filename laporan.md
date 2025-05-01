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
- Menghasilkan sejumlah rekomendasi restoran yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi restoran yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.

### Solution statements
Untuk mencapai tujuan tersebut, digunakan dua pendekatan:
- Content-Based Filtering: Rekomendasi didasarkan pada kesamaan fitur buku (judul, penulis, dll.).
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
      <img width="642" alt="Screenshot 2025-05-01 at 23 42 19" src="https://github.com/user-attachments/assets/309a21d8-c260-4a3a-acab-e720b4315a20" />

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
7. Menggabungkan 

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
