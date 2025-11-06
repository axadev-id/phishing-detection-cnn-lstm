"""
Check all dependencies for improved CNN-LSTM model
"""
print("="*70)
print(" "*15 + "CHECKING ALL DEPENDENCIES")
print("="*70)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   • GPU {i}: {gpu.name}")
    else:
        print("⚠️  No GPU detected (CPU will be used)")
    
except ImportError as e:
    print(f"❌ TensorFlow: NOT INSTALLED - {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: NOT INSTALLED - {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: NOT INSTALLED - {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib: NOT INSTALLED - {e}")

try:
    import seaborn
    print(f"✅ Seaborn: {seaborn.__version__}") # type: ignore
except ImportError as e:
    print(f"❌ Seaborn: NOT INSTALLED - {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: NOT INSTALLED - {e}")

try:
    from imblearn.over_sampling import SMOTE
    import imblearn
    print(f"✅ Imbalanced-learn: {imblearn.__version__}")
except ImportError as e:
    print(f"❌ Imbalanced-learn: NOT INSTALLED - {e}")

try:
    import shap
    print(f"✅ SHAP: {shap.__version__}")
except ImportError as e:
    print(f"❌ SHAP: NOT INSTALLED - {e}")

try:
    from scipy.stats import entropy
    import scipy
    print(f"✅ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"❌ SciPy: NOT INSTALLED - {e}")

print("\n" + "="*70)
print("✅ ALL DEPENDENCIES CHECK COMPLETED!")
print("="*70)

# Test TensorFlow GPU
if gpus:
    print("\n🧪 Testing GPU with sample operation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
        print(f"✅ GPU Test: SUCCESS")
        print(f"✅ Result device: {c.device}")
    except Exception as e:
        print(f"❌ GPU Test: FAILED - {e}")

print("\n" + "="*70)
print("🚀 READY TO RUN IMPROVED CNN-LSTM MODEL!")
print("="*70)
