# SteamGameRecommender

Repository ini berisi kode, dataset, dan dokumentasi untuk Proyek ini memanfaatkan dataset dari Steam dan mengimplementasikan algoritma seperti **Singular Value Decomposition (SVD)** untuk memberikan rekomendasi game yang relevan dan dipersonalisasi.

## Dataset  
Dataset yang digunakan berasal dari Kaggle:  
[Game Recommendations on Steam](https://www.kaggle.com/datasets/bayuabdurrosyidyeye/steam-game-review?select=games.csv).  

## Fitur Utama
- **Rekomendasi Game Berdasarkan Preferensi Pengguna**: Menggunakan algoritma **Collaborative Filtering** dengan **Singular Value Decomposition (SVD)** untuk memberikan rekomendasi game berdasarkan pola preferensi pengguna yang mirip.
- **Matrix Factorization**: Menggunakan teknik matrix factorization untuk mengurangi dimensi data interaksi antara pengguna dan game, sehingga lebih efisien dalam memberikan rekomendasi.
- **Rasio Positif Game**: Menampilkan game berdasarkan rasio positifnya, memberikan rekomendasi game dengan rating yang lebih tinggi untuk pengalaman pengguna yang lebih baik.
- **Genre Game**: Mempertimbangkan genre game untuk memberikan rekomendasi yang relevan dengan minat pengguna.
- **Harga Game**: Menampilkan harga game yang direkomendasikan untuk memberikan pertimbangan berdasarkan anggaran pengguna.

## Struktur Repository
- `/docs`: folder ini berisikan dokumentasi dari project ini
- `/pengembangan`: folder untuk pengembangan versi untuk terminal hanya mencobak bagian backendnya saja untuk algoritma sistem rekomendasi menggunakan collaborative filltering
- `/template_static`: template yang berlum di masukkan dalam flask hanya template static biasa
- `/static`: folder yang berisikan semua komponen frontend dari template yang di gunakan
- `/templates`: folter yang berisikan file html setiap pagenya


## Cara Penggunaan
1. **Clone repository**:  
   ```bash
   git clone https://github.com/username/SteamGameRecommender.git
