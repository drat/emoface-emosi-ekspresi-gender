
# EmoFace: Deteksi Emosi dan Analisis Wajah

![EmoFace Banner](images/emoface-banner.png)

**Pengembang:** _drat

## Deskripsi Proyek

Selamat datang di EmoFace, proyek yang memungkinkan Anda untuk mendeteksi emosi dan melakukan analisis wajah dengan mudah. EmoFace dilengkapi dengan kemampuan mendeteksi wajah, usia, jenis kelamin, dan ekspresi emosi dalam gambar. Ini adalah alat yang berguna untuk pemrosesan gambar dan analisis data dengan berbagai aplikasi seperti penelitian emosi, analisis gambar, dan banyak lagi.

## Fitur Utama

- Mendeteksi wajah dalam gambar.
- Menganalisis usia dan jenis kelamin dari wajah yang terdeteksi.
- Mengenali ekspresi emosi pada wajah.
- Mudah digunakan dengan antarmuka Python yang sederhana.

## Contoh Gambar

### Sebelum Diolah

![Sebelum Diolah](images/sebelum-diolah.png)

### Sesudah Diolah

![Sesudah Diolah](images/sesudah-diolah.png)

## Teknologi yang Digunakan

Proyek ini menggunakan berbagai teknologi termasuk:

- OpenCV: Untuk mendeteksi wajah dan menggambar kotak wajah.
- TensorFlow: Untuk menganalisis ekspresi emosi pada wajah.
- Caffe: Untuk menganalisis usia dan jenis kelamin.
- Keras: Untuk model jaringan saraf tiruan.

## Cara Memulai

1. Pastikan Anda telah menginstal semua dependensi yang dibutuhkan. Anda dapat melihatnya dalam file `requirements.txt`. Unduh file model yang dibutuhkan seperti `age_net.caffemodel` dan `gender_net.caffemodel` di [repositori github ini](https://github.com/GilLevi/AgeGenderDeepLearning/tree/master/models).
2. Jalankan proyek dengan menjalankan script `app.py`.
3. Tunggu hingga proses selesai, hasil analisis akan ditampilkan dalam gambar yang telah diolah.

## Kontribusi

Kami sangat menghargai kontribusi Anda untuk pengembangan proyek ini. Jika Anda ingin berkontribusi, silakan buka permintaan tarik (pull request) dengan perubahan Anda. Pastikan untuk mengikuti panduan kontribusi kami.

## Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE).

---
Dikembangkan oleh _drat.

Terima kasih telah menggunakan EmoFace!
