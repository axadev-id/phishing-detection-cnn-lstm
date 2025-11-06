# 🔥 Panduan Enable GPU untuk TensorFlow di Windows

## 📊 Status Saat Ini

✅ **GPU Terdeteksi:**
- **Model**: NVIDIA GeForce RTX 2050
- **Driver**: 566.07
- **CUDA Version**: 12.7
- **VRAM**: 4 GB
- **Status**: GPU aktif dan berfungsi

❌ **TensorFlow GPU Support:**
- TensorFlow 2.20.0 (CPU only) terinstall
- GPU tidak terdeteksi oleh TensorFlow
- Perlu install CUDA toolkit dan cuDNN

---

## 🚀 Solusi: 3 Opsi

### **Opsi 1: Tetap Gunakan CPU (RECOMMENDED untuk sekarang)**

**Pros:**
- ✅ Sudah siap pakai, tidak perlu install apapun
- ✅ Tidak ada risiko dependency conflict
- ✅ Hasil model sama persis dengan GPU
- ✅ Lebih stabil dan reliable

**Cons:**
- ⏱️ Training lebih lambat (90-120 menit vs 20-40 menit)

**Action:**
```
✅ Tidak perlu action apapun
✅ Langsung jalankan notebook sekarang!
```

---

### **Opsi 2: Install CUDA + cuDNN Manual (ADVANCED)**

**Untuk enable GPU support di Windows, perlu:**

#### Step 1: Install CUDA Toolkit
1. Download: [CUDA Toolkit 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive)
   - Pilih: Windows → x86_64 → 11 → exe (network)
2. Run installer dan ikuti wizard
3. Setelah install, restart komputer

#### Step 2: Install cuDNN
1. Download: [cuDNN 8.9 for CUDA 12.x](https://developer.nvidia.com/cudnn)
   - Perlu login NVIDIA Developer (gratis)
2. Extract zip file
3. Copy files ke CUDA folder:
   ```
   cuDNN/bin/*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin
   cuDNN/include/*.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include
   cuDNN/lib/*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64
   ```

#### Step 3: Set Environment Variables
Add to PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp
```

#### Step 4: Install TensorFlow
```powershell
# Uninstall current TensorFlow
pip uninstall tensorflow

# Install with CUDA (versi yang kompatibel)
pip install tensorflow==2.16.1

# Verify GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Estimated Time:** 1-2 jam (download + install + troubleshooting)

**Risks:**
- ⚠️ Bisa error karena dependency conflict
- ⚠️ Perlu download ~3-4 GB
- ⚠️ Butuh restart komputer
- ⚠️ Mungkin perlu troubleshooting

---

### **Opsi 3: Gunakan Google Colab (EASIEST untuk GPU)**

**Pros:**
- ✅ GPU Tesla T4 gratis (lebih cepat dari RTX 2050)
- ✅ Tidak perlu install apapun
- ✅ TensorFlow + GPU sudah ready
- ✅ 15 GB RAM (lebih banyak dari laptop)
- ✅ Bisa diakses dari mana saja

**Cons:**
- 🌐 Perlu internet connection
- ⏱️ Session timeout setelah 12 jam idle
- 📁 Perlu upload dataset (250 MB)

**Action:**
1. Buka: https://colab.research.google.com
2. Upload notebook: `phishing_detection_cnn_lstm copy1.ipynb`
3. Upload dataset: `dataset/Dataset.csv`
4. Runtime → Change runtime type → GPU (T4 atau V100)
5. Run All Cells

**Estimated Time:** 15-20 menit setup + 15-25 menit training

---

## 💡 Rekomendasi Saya

### **Untuk Sekarang (Penelitian):**
👉 **Gunakan CPU** (Opsi 1)
- Sudah ready, tidak perlu ribet
- Training malam ini (~2 jam)
- Besok pagi hasil sudah jadi
- Paling aman dan reliable

### **Untuk Masa Depan:**
👉 **Setup Google Colab** (Opsi 3)
- Lebih cepat dari RTX 2050
- Tidak perlu install CUDA/cuDNN
- Bisa untuk training ulang atau eksperimen

### **Jika Ingin GPU Lokal:**
👉 **Setup CUDA Manual** (Opsi 2)
- Hanya jika punya waktu 1-2 jam
- Setelah penelitian selesai
- Untuk long-term development

---

## 🎯 Action Plan yang Saya Rekomendasikan

### **HARI INI (6 Nov 2025, 19:50):**

```powershell
# 1. Jalankan notebook dengan CPU (SEKARANG!)
jupyter notebook
# Buka: phishing_detection_cnn_lstm copy1.ipynb
# Run All Cells
# Training time: ~90-120 menit
```

**Timeline:**
- 19:50 - 20:00: Setup & start training
- 20:00 - 22:00: Training berjalan (biarkan laptop jangan dimatikan)
- 22:00 - 22:30: Evaluasi hasil & export model
- **Result:** Model trained dengan accuracy ~95%+

### **BESOK (Jika ingin GPU):**

**Opsi A: Google Colab**
```
1. Upload notebook ke Colab
2. Upload dataset
3. Select GPU runtime
4. Run training (15-25 menit)
5. Download model
```

**Opsi B: Local GPU Setup**
```
1. Install CUDA Toolkit (30 min)
2. Install cuDNN (15 min)
3. Setup environment variables (10 min)
4. Install TensorFlow with GPU (10 min)
5. Troubleshooting jika ada error (0-60 min)
6. Run training (20-30 menit)
```

---

## 📊 Performance Comparison

| Device | Training Time | Setup Time | Complexity |
|--------|--------------|------------|------------|
| **CPU** | 90-120 min | ✅ 0 min | ⭐ Easy |
| **RTX 2050 (Local)** | 30-40 min | ⚠️ 60-120 min | ⭐⭐⭐⭐ Hard |
| **Colab GPU (T4)** | 15-25 min | ✅ 10 min | ⭐⭐ Medium |

---

## ✅ Current Status Summary

**Environment:**
- ✅ Python 3.10.18
- ✅ TensorFlow 2.20.0 (CPU)
- ✅ All dependencies installed
- ✅ GPU hardware available (RTX 2050)
- ❌ GPU not accessible by TensorFlow

**Ready to Train:**
- ✅ Notebook ready
- ✅ Dataset ready
- ✅ All code working
- ⏱️ Estimated training time: 90-120 minutes

**Recommendation:**
```
🚀 START TRAINING NOW with CPU!
🔧 Setup GPU support later (optional)
```

---

## 🆘 Need Help?

**Jika ingin saya bantu:**
1. **Start training CPU** → Say: "mulai training sekarang"
2. **Setup Colab** → Say: "bantu setup google colab"
3. **Install CUDA** → Say: "guide install cuda step by step"

**Yang mana yang ingin Anda lakukan sekarang?** 😊
