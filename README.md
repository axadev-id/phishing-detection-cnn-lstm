# Deteksi Phishing dengan CNN + LSTM Hybrid Model

## 📋 Deskripsi Proyek

Proyek penelitian Tugas Akhir untuk deteksi URL phishing menggunakan arsitektur deep learning hybrid yang menggabungkan Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM). Model ini menganalisis 41 fitur berbasis URL untuk mengklasifikasikan apakah sebuah URL merupakan legitimate atau phishing.

## 🎯 Tujuan

- Mengembangkan model deep learning yang akurat untuk mendeteksi phishing
- Memanfaatkan kekuatan CNN untuk ekstraksi fitur lokal
- Menggunakan LSTM untuk menangkap dependensi sekuensial dalam data
- Mencapai performa tinggi dalam klasifikasi phishing vs legitimate

## 🏗️ Arsitektur Model

### CNN Layers
- **Conv1D Layer 1**: 64 filters, kernel size 3
- **Conv1D Layer 2**: 128 filters, kernel size 3
- **Conv1D Layer 3**: 256 filters, kernel size 3
- MaxPooling1D dan BatchNormalization

### LSTM Layers
- **LSTM Layer 1**: 128 units (return sequences)
- **LSTM Layer 2**: 64 units

### Dense Layers
- Dense 64 units (ReLU)
- Dense 32 units (ReLU)
- Output: 1 unit (Sigmoid) untuk binary classification

### Regularization
- Dropout layers (0.2 - 0.4)
- BatchNormalization
- Class weights untuk handling imbalanced data

## 📊 Dataset

- **Total Sampel**: 247,952 URL
- **Fitur**: 41 fitur berbasis URL
- **Target**: 
  - 0 = Legitimate
  - 1 = Phishing
- **Lokasi**: `dataset/Dataset.csv`

### Fitur-fitur URL:
- URL length, number of dots, hyphens, special characters
- Domain characteristics
- Subdomain properties
- Path and query features
- Entropy metrics

## 📁 Struktur Proyek

```
projek_phishing/
├── dataset/
│   └── Dataset.csv              # Dataset phishing
├── models/
│   ├── best_cnn_lstm_model.h5   # Model terbaik (setelah training)
│   ├── scaler.pkl               # Scaler untuk normalisasi
│   └── model_summary.txt        # Summary arsitektur model
├── results/
│   ├── training_history.png     # Visualisasi training
│   ├── confusion_matrix.png     # Confusion matrix
│   ├── roc_curve.png            # ROC curve
│   ├── probability_distribution.png
│   ├── performance_comparison.png
│   ├── training_history.json    # History dalam JSON
│   └── evaluation_results.json  # Hasil evaluasi
├── logs/
│   └── fit/                     # TensorBoard logs
├── phishing_detection_cnn_lstm.ipynb  # Notebook utama penelitian
├── requirements.txt             # Dependencies
└── README.md                    # Dokumentasi ini
```

## 🚀 Setup dan Installation

### 1. Clone atau Download Repository

```bash
cd d:\kuliah\TA\Phising\projek_phishing
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verifikasi Instalasi

```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 💻 Cara Penggunaan

### Opsi 1: Menggunakan Jupyter Notebook (Recommended)

Notebook sudah berisi semua langkah dari preprocessing hingga evaluasi dengan visualisasi lengkap.

```powershell
jupyter lab
# atau
jupyter notebook
```

Buka file: `phishing_detection_cnn_lstm.ipynb`

Jalankan semua cell secara berurutan untuk:
1. Load dan eksplorasi dataset
2. Preprocessing data
3. Membangun model CNN + LSTM
4. Training model
5. Evaluasi dan visualisasi hasil

**Note**: Semua kode penelitian sudah terintegrasi dalam notebook, dari data loading hingga evaluasi final.

## 📈 Training Model

Model menggunakan callbacks untuk optimalisasi training:

- **Early Stopping**: Patience 15 epochs
- **Model Checkpoint**: Simpan model terbaik berdasarkan validation accuracy
- **Reduce Learning Rate**: Factor 0.5, patience 7 epochs
- **TensorBoard**: Logging untuk visualisasi

```python
# Hyperparameters default
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
```

## 📊 Monitoring Training dengan TensorBoard

```powershell
tensorboard --logdir=logs/fit
```

Buka browser di: http://localhost:6006

## 🎯 Evaluasi Model

Model dievaluasi menggunakan berbagai metrik:

- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Akurasi prediksi phishing
- **Recall**: Kemampuan mendeteksi semua phishing
- **F1-Score**: Harmonic mean precision dan recall
- **AUC-ROC**: Area under ROC curve

### Visualisasi Hasil:
- Confusion Matrix
- ROC Curve
- Training History (Loss, Accuracy, Precision, Recall)
- Probability Distribution
- Performance Comparison

## 📊 Hasil Model (Test Set)

Setelah training dengan dataset 247,952 sampel, model CNN + LSTM Hybrid mencapai performa sebagai berikut:

### Metrik Performa

| Metrik | Nilai | Persentase |
|--------|-------|------------|
| **Accuracy** | 0.9217 | **92.17%** |
| **Precision** | 0.9348 | **93.48%** |
| **Recall** | 0.9001 | **90.01%** |
| **F1-Score** | 0.9171 | **91.71%** |
| **AUC-ROC** | 0.9749 | **97.49%** |

### Confusion Matrix Detail

| Metrik | Jumlah | Keterangan |
|--------|--------|------------|
| **True Negative (TN)** | 24,209 | Legitimate diprediksi Legitimate ✅ |
| **False Positive (FP)** | 1,499 | Legitimate diprediksi Phishing ⚠️ |
| **False Negative (FN)** | 2,386 | Phishing diprediksi Legitimate ⚠️ |
| **True Positive (TP)** | 21,496 | Phishing diprediksi Phishing ✅ |

### Analisis Hasil

✅ **Kekuatan Model:**
- **Precision tinggi (93.48%)**: Model sangat akurat dalam mengidentifikasi phishing dengan tingkat false positive rendah
- **AUC-ROC sangat baik (97.49%)**: Model memiliki kemampuan pemisahan kelas yang excellent
- **Accuracy solid (92.17%)**: Performa keseluruhan sangat baik pada data test
- **Balanced Performance**: F1-Score 91.71% menunjukkan keseimbangan antara precision dan recall

⚠️ **Kelemahan Model:**
- **Recall 90.01%**: Masih ada 2,386 URL phishing yang lolos dari deteksi (False Negative)
- **False Positive 1,499**: Beberapa legitimate URL salah diklasifikasikan sebagai phishing
- **Trade-off**: Model lebih condong ke precision daripada recall

### Interpretasi Bisnis

🎯 **Untuk Sistem Keamanan:**
- Model dapat mendeteksi **90% dari semua phishing** (Recall)
- Ketika model mengatakan "Phishing", **93.5% benar** (Precision)
- Cocok untuk **sistem warning/filtering** dengan sedikit false alarm
- **10% phishing** masih bisa lolos (perlu layer keamanan tambahan)

## 💡 Saran Peningkatan Model

### 1. Peningkatan Arsitektur

#### A. Attention Mechanism
```python
# Tambahkan attention layer setelah LSTM
from tensorflow.keras.layers import Attention, Concatenate

# Attention untuk fokus pada fitur penting
attention_layer = Attention()([lstm_output, lstm_output])
```
**Manfaat**: Meningkatkan recall dengan fokus pada fitur phishing penting

#### B. Bidirectional LSTM
```python
from tensorflow.keras.layers import Bidirectional

# Ganti LSTM dengan Bidirectional LSTM
model.add(Bidirectional(LSTM(128, return_sequences=True)))
```
**Manfaat**: Menangkap dependensi forward dan backward, potensi +2-3% accuracy

#### C. Residual Connections (ResNet-style)
```python
# Skip connections untuk deep network
x = Conv1D(...)(input)
residual = x
x = Conv1D(...)(x)
x = Add()([x, residual])
```
**Manfaat**: Training lebih stabil, gradient flow lebih baik

### 2. Data Augmentation & Feature Engineering

#### A. Feature Selection
- Analisis feature importance menggunakan **SHAP** atau **LIME**
- Hapus fitur redundant, fokus pada fitur dengan korelasi tinggi
- **Expected improvement**: +1-2% accuracy, training lebih cepat

#### B. Feature Engineering
```python
# Tambahkan fitur kombinasi
df['url_length_ratio'] = df['url_length'] / df['domain_length']
df['special_char_density'] = df['num_special_chars'] / df['url_length']
df['entropy_normalized'] = df['entropy'] / np.log2(df['url_length'])
```
**Manfaat**: Fitur baru bisa menangkap pola yang tidak terlihat

#### C. SMOTE untuk Imbalanced Data
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```
**Manfaat**: Mengatasi imbalance, meningkatkan recall

### 3. Ensemble Methods

#### A. Model Stacking
```python
# Gabungkan CNN+LSTM dengan model lain
# Model 1: CNN+LSTM
# Model 2: Random Forest
# Model 3: XGBoost
# Meta-learner: Logistic Regression

ensemble_pred = (cnn_lstm_pred + rf_pred + xgb_pred) / 3
```
**Expected improvement**: +2-4% accuracy, lebih robust

#### B. Voting Classifier
- Kombinasi 3-5 model dengan voting
- Hard voting untuk classification
- Soft voting untuk probabilistic
**Manfaat**: Mengurangi False Negative dan False Positive

### 4. Hyperparameter Tuning

#### A. Learning Rate Scheduling
```python
# Cyclical Learning Rate
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = LearningRateScheduler(scheduler)
```

#### B. Grid Search / Bayesian Optimization
```python
# Tune hyperparameters
param_grid = {
    'lstm_units': [[128, 64], [256, 128], [128, 128]],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128]
}
```
**Expected improvement**: +1-3% accuracy

### 5. Post-Processing & Threshold Tuning

#### A. Optimal Threshold
```python
# Cari threshold optimal (default 0.5)
# Untuk maximize F1-Score atau balance precision-recall

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
# Plot dan pilih threshold optimal
optimal_threshold = 0.45  # contoh hasil analisis
```
**Manfaat**: Sesuaikan threshold untuk use case (security vs user experience)

#### B. Confidence-based Filtering
```python
# High confidence: Langsung action
# Medium confidence: Manual review
# Low confidence: Whitelist

if pred_proba > 0.8:
    action = "Block"
elif pred_proba > 0.5:
    action = "Review"
else:
    action = "Allow"
```

### 6. Transfer Learning & Pre-trained Models

- Gunakan **BERT** atau **transformer** untuk URL text embedding
- Pre-trained pada dataset phishing besar (PhishTank, OpenPhish)
- Fine-tune pada dataset spesifik
**Expected improvement**: +3-5% accuracy, especially on new phishing patterns

### 7. Real-time Learning & Adaptive Model

```python
# Implement online learning
# Model di-update dengan data baru secara berkala

from river import tree

# Incremental learning
model.partial_fit(new_data, new_labels)
```
**Manfaat**: Model tetap update dengan phishing terbaru

### 8. Explainability & Interpretability

```python
# SHAP untuk explain predictions
import shap

explainer = shap.DeepExplainer(model, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample)
```
**Manfaat**: 
- Understand why model predict phishing
- Trust & transparency
- Debugging false predictions

### Prioritas Implementasi

**🔥 High Priority (Quick Wins):**
1. Threshold Tuning (no retraining needed) → +0.5-1% improvement
2. Feature Engineering → +1-2% improvement
3. Bidirectional LSTM → +2-3% improvement

**🌟 Medium Priority (Moderate Effort):**
4. Attention Mechanism → +2-4% improvement
5. Hyperparameter Tuning → +1-3% improvement
6. SMOTE / Data Balancing → +1-2% recall

**🚀 Long-term (High Effort, High Reward):**
7. Ensemble Methods → +2-4% improvement
8. Transfer Learning (BERT) → +3-5% improvement
9. Real-time Learning System → Continuous improvement

### Target Performa

Dengan implementasi saran di atas, target realistis:

| Metrik | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Accuracy | 92.17% | **95-96%** | +3-4% |
| Precision | 93.48% | **95-97%** | +2-3% |
| Recall | 90.01% | **93-95%** | +3-5% |
| F1-Score | 91.71% | **94-96%** | +2-4% |
| AUC-ROC | 97.49% | **98-99%** | +1-2% |

**🎯 Goal**: Mencapai **95%+ accuracy** dengan **recall >93%** untuk mengurangi False Negative

## 🔮 Prediksi URL Baru

Setelah model ditraining, gunakan untuk prediksi:

```python
import pickle
from tensorflow import keras
import numpy as np

# Load model dan scaler
model = keras.models.load_model('models/best_cnn_lstm_model.h5')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Siapkan fitur URL baru (41 fitur)
new_url_features = np.array([[...]])  # 41 fitur

# Preprocessing
new_url_scaled = scaler.transform(new_url_features)
new_url_reshaped = new_url_scaled.reshape(1, 41, 1)

# Prediksi
prediction_proba = model.predict(new_url_reshaped)
prediction = 1 if prediction_proba[0][0] > 0.5 else 0

print(f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
print(f"Confidence: {prediction_proba[0][0]:.4f}")
```

## 🛠️ Konfigurasi

Untuk mengubah hyperparameters, edit langsung di notebook `phishing_detection_cnn_lstm.ipynb`:

```python
# Data parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

# Model hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# CNN parameters
CNN_FILTERS = [64, 128, 256]
KERNEL_SIZE = 3

# LSTM parameters
LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.3
```

Semua konfigurasi dapat disesuaikan langsung di cell-cell notebook yang relevan.

## 📝 Requirements

- Python >= 3.8
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## 🐛 Troubleshooting

### GPU tidak terdeteksi
```powershell
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu
```

### Memory Error saat Training
```python
# Kurangi batch size di config.py
BATCH_SIZE = 32  # atau 16
```

### Import Error
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📚 Referensi

- Dataset: URL-based phishing detection features
- Deep Learning: CNN + LSTM Hybrid Architecture
- Framework: TensorFlow/Keras
- Preprocessing: StandardScaler, train-test split

## 👨‍💻 Author

Proyek Tugas Akhir - Deteksi Phishing

## 📄 License

Untuk keperluan penelitian akademik.

## 🙏 Acknowledgments

- Dataset provider
- TensorFlow/Keras documentation
- Scikit-learn library

---

## 📞 Support

Untuk pertanyaan atau issues, silakan hubungi melalui:
- Email: [your-email]
- Repository: [your-repo-link]

---

**Note**: Pastikan untuk menjalankan notebook secara berurutan untuk hasil optimal. Model terbaik akan disimpan secara otomatis selama training.

🎉 **Happy Researching!**
