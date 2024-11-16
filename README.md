# SteamGameRecommender

Repository ini berisi kode, dataset, dan dokumentasi untuk penelitian sistem rekomendasi game menggunakan metode **Collaborative Filtering**. Proyek ini memanfaatkan dataset dari Steam dan mengimplementasikan algoritma seperti **Singular Value Decomposition (SVD)** dan **Pearson Correlation** untuk memberikan rekomendasi game yang relevan dan dipersonalisasi.

## Dataset  
Dataset yang digunakan berasal dari Kaggle:  
[Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data).  
Dataset ini mencakup:
- **games.csv**: Informasi tentang game, termasuk harga, rating, dan tanggal rilis.
- **users.csv**: Informasi publik profil pengguna, seperti jumlah pembelian dan ulasan yang dipublikasikan.
- **recommendations.csv**: Data ulasan pengguna, termasuk apakah mereka merekomendasikan game tertentu.

## Fitur Utama
- **Pra-pemrosesan Data**: Membersihkan dan menyiapkan dataset untuk digunakan dalam model rekomendasi.
- **Implementasi Model**: 
  - **User-Based Collaborative Filtering** dengan Pearson Correlation.
  - **Model-Based Collaborative Filtering** dengan SVD.
- **Evaluasi Kinerja**: Menggunakan metrik seperti MAE dan RMSE untuk mengukur akurasi model.

## Struktur Repository
- `/src`: Kode Python untuk pemrosesan data dan pengembangan model rekomendasi.
- `/docs`: Dokumentasi dan laporan penelitian.

## Cara Penggunaan
1. **Clone repository**:  
   ```bash
   git clone https://github.com/username/SteamGameRecommender.git
