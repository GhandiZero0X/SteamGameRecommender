# SteamGameRecommender

Repository ini berisi kode, dataset, dan dokumentasi untuk Proyek ini memanfaatkan dataset dari Steam dan mengimplementasikan algoritma seperti **Singular Value Decomposition (SVD)** untuk memberikan rekomendasi game yang relevan dan dipersonalisasi.

## Dataset  
Dataset yang digunakan berasal dari Kaggle:  
[Game Recommendations on Steam](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings/data?select=steam_game_reviews.csv).  

## Fitur Utama
- **Pra-pemrosesan Data**: Membersihkan dan menyiapkan dataset untuk digunakan dalam model rekomendasi.
- **Implementasi Model**: 
  - **Model-Based Collaborative Filtering** dengan SVD.
- **Evaluasi Kinerja**: Menggunakan metrik seperti MAE dan RMSE untuk mengukur akurasi model.

## Struktur Repository
- `/src`: Kode Python untuk pemrosesan data dan pengembangan model rekomendasi.
- `/docs`: Dokumentasi dan laporan penelitian.

## Cara Penggunaan
1. **Clone repository**:  
   ```bash
   git clone https://github.com/username/SteamGameRecommender.git
