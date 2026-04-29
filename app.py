from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carica il modello e i dati necessari
# NOTA: devi prima esportare il modello dal notebook con:
#   import pickle
#   with open('model.pkl', 'wb') as f:
#       pickle.dump({'model': rs_xgb, 'genre_mean': genre_mean, 'columns': X_train.columns.tolist()}, f)
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    genre_mean = data['genre_mean']
    columns = data['columns']

GENRES = sorted(genre_mean.index.tolist())

HTML = """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spotify Popularity Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        font-family: 'DM Sans', sans-serif;
        background: #0d0d0d;
        color: #e0e0e0;
        min-height: 100vh;
        padding: 40px 20px;
    }

    .container {
        max-width: 800px;
        margin: 0 auto;
    }

    h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: #1DB954;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 40px;
    }

    .section-title {
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #1DB954;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 1px solid #222;
    }

    .slider-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px 32px;
        margin-bottom: 32px;
    }

    .slider-group {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .slider-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }

    .slider-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #ccc;
    }

    .slider-value {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #1DB954;
        min-width: 50px;
        text-align: right;
    }

    input[type="range"] {
        -webkit-appearance: none;
        width: 100%;
        height: 4px;
        border-radius: 2px;
        background: #333;
        outline: none;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #1DB954;
        cursor: pointer;
        transition: transform 0.15s;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
        transform: scale(1.3);
    }

    .select-group {
        margin-bottom: 32px;
    }

    select {
        width: 100%;
        padding: 10px 14px;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        color: #e0e0e0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        outline: none;
        cursor: pointer;
    }

    select:focus { border-color: #1DB954; }

    .checkbox-group {
        display: flex;
        gap: 32px;
        margin-bottom: 32px;
    }

    .checkbox-item {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
    }

    .checkbox-item input { accent-color: #1DB954; width: 16px; height: 16px; }

    .predict-btn {
        width: 100%;
        padding: 14px;
        background: #1DB954;
        color: #000;
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: background 0.2s, transform 0.1s;
        letter-spacing: 1px;
    }

    .predict-btn:hover { background: #1ed760; transform: translateY(-1px); }
    .predict-btn:active { transform: translateY(0); }

    .result-box {
        margin-top: 28px;
        padding: 28px;
        background: #141414;
        border: 1px solid #222;
        border-radius: 12px;
        text-align: center;
        display: none;
    }

    .result-score {
        font-family: 'Space Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
    }

    .result-label {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 16px;
    }

    .result-bar-bg {
        width: 100%;
        height: 8px;
        background: #222;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .result-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    .result-verdict {
        font-weight: 500;
        font-size: 1rem;
    }

    @media (max-width: 600px) {
        .slider-grid { grid-template-columns: 1fr; }
        h1 { font-size: 1.5rem; }
    }
</style>
</head>
<body>
<div class="container">
    <h1>&#127925; Popularity Predictor</h1>
    <p class="subtitle">Stima la popolarità di un brano Spotify a partire dalle sue feature audio</p>

    <div class="section-title">Genere musicale</div>
    <div class="select-group">
        <select id="track_genre">
            {% for g in genres %}
            <option value="{{ g }}" {{ 'selected' if g == 'pop' }}>{{ g }}</option>
            {% endfor %}
        </select>
    </div>

    <div class="section-title">Feature audio</div>
    <div class="slider-grid">
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Danceability</span>
                <span class="slider-value" id="val_danceability">0.50</span>
            </div>
            <input type="range" id="danceability" min="0" max="1" step="0.01" value="0.50">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Energy</span>
                <span class="slider-value" id="val_energy">0.50</span>
            </div>
            <input type="range" id="energy" min="0" max="1" step="0.01" value="0.50">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Loudness (dB)</span>
                <span class="slider-value" id="val_loudness">-10.0</span>
            </div>
            <input type="range" id="loudness" min="-60" max="0" step="0.1" value="-10">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Speechiness</span>
                <span class="slider-value" id="val_speechiness">0.05</span>
            </div>
            <input type="range" id="speechiness" min="0" max="1" step="0.01" value="0.05">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Acousticness</span>
                <span class="slider-value" id="val_acousticness">0.50</span>
            </div>
            <input type="range" id="acousticness" min="0" max="1" step="0.01" value="0.50">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Instrumentalness</span>
                <span class="slider-value" id="val_instrumentalness">0.00</span>
            </div>
            <input type="range" id="instrumentalness" min="0" max="1" step="0.01" value="0.00">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Liveness</span>
                <span class="slider-value" id="val_liveness">0.10</span>
            </div>
            <input type="range" id="liveness" min="0" max="1" step="0.01" value="0.10">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Valence</span>
                <span class="slider-value" id="val_valence">0.50</span>
            </div>
            <input type="range" id="valence" min="0" max="1" step="0.01" value="0.50">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Tempo (BPM)</span>
                <span class="slider-value" id="val_tempo">120</span>
            </div>
            <input type="range" id="tempo" min="30" max="250" step="1" value="120">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Durata (secondi)</span>
                <span class="slider-value" id="val_duration_s">200</span>
            </div>
            <input type="range" id="duration_s" min="30" max="600" step="1" value="200">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Key (tonalità)</span>
                <span class="slider-value" id="val_key">0</span>
            </div>
            <input type="range" id="key" min="0" max="11" step="1" value="0">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Time Signature</span>
                <span class="slider-value" id="val_time_signature">4</span>
            </div>
            <input type="range" id="time_signature" min="3" max="7" step="1" value="4">
        </div>
        <div class="slider-group">
            <div class="slider-header">
                <span class="slider-label">Num. artisti</span>
                <span class="slider-value" id="val_num_artists">1</span>
            </div>
            <input type="range" id="num_artists" min="1" max="10" step="1" value="1">
        </div>
    </div>

    <div class="section-title">Altre opzioni</div>
    <div class="checkbox-group">
        <label class="checkbox-item">
            <input type="checkbox" id="explicit">
            <span class="slider-label">Explicit</span>
        </label>
        <label class="checkbox-item">
            <input type="checkbox" id="mode" checked>
            <span class="slider-label">Modo maggiore</span>
        </label>
    </div>

    <button class="predict-btn" onclick="predict()">PREDICI POPOLARITÀ</button>

    <div class="result-box" id="resultBox">
        <div class="result-score" id="resultScore">—</div>
        <div class="result-label">su 100</div>
        <div class="result-bar-bg">
            <div class="result-bar-fill" id="resultBar"></div>
        </div>
        <div class="result-verdict" id="resultVerdict"></div>
    </div>
</div>

<script>
    // Aggiorna i valori visualizzati degli slider
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', () => {
            const valSpan = document.getElementById('val_' + slider.id);
            if (valSpan) valSpan.textContent = slider.value;
        });
    });

    async function predict() {
        const features = {
            danceability: parseFloat(document.getElementById('danceability').value),
            energy: parseFloat(document.getElementById('energy').value),
            key: parseInt(document.getElementById('key').value),
            loudness: parseFloat(document.getElementById('loudness').value),
            mode: document.getElementById('mode').checked ? 1 : 0,
            speechiness: parseFloat(document.getElementById('speechiness').value),
            acousticness: parseFloat(document.getElementById('acousticness').value),
            instrumentalness: parseFloat(document.getElementById('instrumentalness').value),
            liveness: parseFloat(document.getElementById('liveness').value),
            valence: parseFloat(document.getElementById('valence').value),
            tempo: parseFloat(document.getElementById('tempo').value),
            duration_s: parseFloat(document.getElementById('duration_s').value),
            explicit: document.getElementById('explicit').checked ? 1 : 0,
            num_artists: parseInt(document.getElementById('num_artists').value),
            time_signature: parseInt(document.getElementById('time_signature').value),
            track_genre: document.getElementById('track_genre').value
        };

        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(features)
        });
        const data = await res.json();
        const score = data.popularity;

        const box = document.getElementById('resultBox');
        const scoreEl = document.getElementById('resultScore');
        const bar = document.getElementById('resultBar');
        const verdict = document.getElementById('resultVerdict');

        box.style.display = 'block';
        scoreEl.textContent = score.toFixed(1);

        let color;
        let text;
        if (score >= 60) {
            color = '#1DB954'; text = '🟢 Brano probabilmente popolare';
        } else if (score >= 40) {
            color = '#F5A623'; text = '🟡 Brano mediamente popolare';
        } else {
            color = '#E63946'; text = '🔴 Brano probabilmente di nicchia';
        }

        scoreEl.style.color = color;
        bar.style.width = score + '%';
        bar.style.background = color;
        verdict.textContent = text;
        verdict.style.color = color;
    }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, genres=GENRES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_dict = {
        'danceability': data['danceability'],
        'energy': data['energy'],
        'key': data['key'],
        'loudness': data['loudness'],
        'mode': data['mode'],
        'speechiness': data['speechiness'],
        'acousticness': data['acousticness'],
        'instrumentalness': data['instrumentalness'],
        'liveness': data['liveness'],
        'valence': data['valence'],
        'tempo': data['tempo'],
        'duration_s': data['duration_s'],
        'explicit': data['explicit'],
        'num_artists': data['num_artists'],
        'time_signature': data['time_signature'],
    }

    input_df = pd.DataFrame([input_dict])

    # Target encoding del genere
    genre = data['track_genre']
    if genre in genre_mean.index:
        input_df['genre_popularity_mean'] = genre_mean[genre]
    else:
        input_df['genre_popularity_mean'] = genre_mean.mean()

    # Allinea le colonne
    input_df = input_df.reindex(columns=columns, fill_value=0)

    pred = model.predict(input_df)[0]
    pred = float(np.clip(pred, 0, 100))

    return jsonify({'popularity': pred})

if __name__ == '__main__':
    app.run(debug=True, port=5000)