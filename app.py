from flask import Flask, render_template, request
from sistem_rekomendasi import get_recommendations_for_user
import pandas as pd

app = Flask(__name__)

def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

recommendations_df = reduce_memory(pd.read_csv('recommendations.csv'))

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk form
@app.route('/form')
def form():
    # Mengambil ID user yang memiliki lebih dari 7 rekomendasi
    user_counts = recommendations_df['user_id'].value_counts()
    recommended_user_ids = user_counts[user_counts >= 7].index.tolist()  # ID user yang memenuhi kriteria
    
    return render_template('form.html', recommended_user_ids=recommended_user_ids)

# Route untuk hasil rekomendasi
@app.route('/result', methods=['POST'])
def result():
    # Ambil ID user dari form
    user_id = int(request.form.get('user_id'))
    
    # Ambil rekomendasi game dari fungsi backend
    recommendations = get_recommendations_for_user(user_id, n_neighbors=6)

    # Render hasil ke halaman result.html
    return render_template('result.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
