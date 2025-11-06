# Quick Start Guide - Deteksi Phishing CNN + LSTM

## 🚀 Langkah Cepat Memulai

### 1. Setup Environment

```powershell
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Notebook

```powershell
# Start Jupyter
jupyter lab

# Atau gunakan Jupyter Notebook
jupyter notebook
```

Buka file: **`phishing_detection_cnn_lstm.ipynb`**

### 3. Jalankan Cell-by-Cell

Notebook sudah berisi semua langkah:
1. ✅ Import libraries
2. ✅ Load dan eksplorasi data
3. ✅ Preprocessing
4. ✅ Build model CNN + LSTM
5. ✅ Training (akan memakan waktu)
6. ✅ Evaluasi dan visualisasi

### 4. Tunggu Training Selesai

Training akan berjalan dengan:
- Early stopping (berhenti otomatis jika tidak improve)
- Model checkpoint (save model terbaik)
- Progress bar untuk monitoring

Estimasi waktu: 30-60 menit (tergantung hardware)

### 5. Lihat Hasil

Setelah training selesai, check folder:
- `models/` - Model dan scaler tersimpan
- `results/` - Visualisasi dan metrics
- `logs/` - TensorBoard logs

## 📊 Monitoring dengan TensorBoard (Opsional)

```powershell
# Di terminal baru
tensorboard --logdir=logs/fit
```

Buka browser: http://localhost:6006

## 🔮 Gunakan Model untuk Prediksi

```python
from src.predictor import PhishingDetector
import numpy as np

# Load model
detector = PhishingDetector()
detector.load_model()

# Prediksi (dengan 41 fitur URL)
features = np.array([...])  # 41 fitur
result = detector.predict_single(features)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## ⚠️ Troubleshooting

### GPU tidak terdeteksi?
- Install CUDA toolkit
- Atau jalankan di CPU (lebih lambat tapi tetap bisa)

### Memory error?
- Kurangi `BATCH_SIZE` di `src/config.py`
- Restart kernel notebook

### Import error?
```powershell
pip install --upgrade tensorflow keras
```

## 💡 Tips

1. **Pertama kali**: Jalankan notebook lengkap dari awal sampai akhir
2. **Monitoring**: Lihat metrics di notebook untuk memastikan model belajar
3. **Save results**: Semua hasil otomatis tersimpan
4. **Experiment**: Coba ubah hyperparameters di `src/config.py`

## 📞 Need Help?

Check `README.md` untuk dokumentasi lengkap!

---

**Happy Coding! 🎉**
