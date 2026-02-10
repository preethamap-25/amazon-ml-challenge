# 🚀 INSTALLATION & USAGE GUIDE

## Quick Installation (3 commands)

```bash
# 1. Install dependencies
pip install pandas numpy torch torchvision Pillow scikit-learn xgboost tqdm requests

# 2. Verify setup
python setup_check.py

# 3. Run solution
python train_and_predict.py
```

---

## 📋 Step-by-Step Instructions

### STEP 1: Setup Environment

```bash
# Navigate to project folder
cd ml_challenge_2025

# Install all dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed pandas-2.0.3 numpy-1.24.3 torch-2.0.1 ...
```

---

### STEP 2: Verify Installation

```bash
python setup_check.py
```

**Expected output:**
```
✓ Python 3.x.x
✓ pandas
✓ numpy
✓ torch
...
✓ All checks passed!
```

If you see errors, install missing packages:
```bash
pip install <package-name>
```

---

### STEP 3: Prepare Your Data

Place your CSV files in the `dataset/` folder:

```
ml_challenge_2025/
└── dataset/
    ├── train.csv      ← Your training data
    └── test.csv       ← Your test data
```

---

### STEP 4: Download Images

```bash
python download_images.py
```

**What happens:**
- Downloads images from URLs in train.csv and test.csv
- Saves to `images/` folder
- Shows progress bar
- Takes 30-60 minutes

**Expected output:**
```
[1/2] Downloading TRAINING images...
100%|████████████| 75000/75000
Successfully downloaded: 73,456

[2/2] Downloading TEST images...
100%|████████████| 75000/75000
Successfully downloaded: 73,892

✓ Image download complete!
```

**Note:** It's normal if 2-5% of images fail to download. The model handles this.

---

### STEP 5: Train Model & Generate Predictions

```bash
python train_and_predict.py
```

**What happens:**
1. Loads training data (75K samples)
2. Extracts image features using ResNet50
3. Extracts text features
4. Trains XGBoost model (500 trees)
5. Generates predictions on test data
6. Saves `test_out.csv`

**Expected output:**
```
==========================================
ML CHALLENGE 2025 - PRICE PREDICTION
==========================================

Loading data from dataset/train.csv
Loaded 75000 samples

FEATURE EXTRACTION
Extracting features from 75000 samples...
Loading pre-trained ResNet50 model...
Extracting features from 75000 images...
100%|████████████| 75000/75000
Total features: 2055

Splitting data: 80% train, 20% validation
Training samples: 60000
Validation samples: 15000

MODEL TRAINING
Training XGBoost model...
[0]  validation_0-rmse:XX.XXX
[50] validation_0-rmse:XX.XXX
...

Training SMAPE: 0.1821
Validation SMAPE: 0.2156

✓ Model saved to models/price_predictor.pkl

GENERATING PREDICTIONS ON TEST DATA
Extracting features from test data...
Making predictions...

✓ Predictions saved to: test_out.csv
Total predictions: 75000
Price range: $1.23 - $9,999.99
Mean price: $89.45

✓ Pipeline completed successfully!
```

**Duration:** 1-2 hours (with GPU) or 3-4 hours (CPU only)

---

### STEP 6: Verify Output

```bash
# Check output file
head test_out.csv
```

**Expected format:**
```csv
sample_id,price
1,29.99
2,149.99
3,12.49
...
```

**Validation checks:**
- ✅ Exactly 75,000 rows
- ✅ Two columns: sample_id, price
- ✅ All prices positive
- ✅ Matches test.csv sample_ids

---

### STEP 7: Submit

Upload `test_out.csv` to the competition portal!

Also submit `Documentation.md` (fill in your team details first).

---

## 🔄 Alternative: Text-Only (Faster)

If you have issues with images or want faster results:

```bash
python train_text_only.py
```

**Advantages:**
- ✓ No image download needed
- ✓ Faster training (30 minutes)
- ✓ Less memory required (8GB RAM ok)
- ✓ CPU-friendly

**Trade-off:**
- Slightly lower accuracy (~5% worse SMAPE)

---

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision
```

### Issue 2: "CUDA out of memory"

**Solution 1:** Force CPU usage
Edit `train_and_predict.py`, line 17:
```python
DEVICE = 'cpu'
```

**Solution 2:** Reduce batch size
Edit line 16:
```python
BATCH_SIZE = 16  # or even 8
```

### Issue 3: Images not downloading

**Causes:**
- Internet connection issues
- Some URLs may be broken (normal)

**Solutions:**
- Retry: `python download_images.py`
- Continue anyway - model handles missing images
- Use text-only: `python train_text_only.py`

### Issue 4: Training too slow

**Solutions:**
1. Use GPU (much faster)
2. Reduce trees: `n_estimators=300`
3. Use text-only model
4. Try Google Colab (free GPU)

### Issue 5: Wrong number of predictions

**Check:**
```bash
wc -l test_out.csv  # Should be 75001 (header + 75000 rows)
```

If wrong, model may have crashed. Check error messages.

---

## 💡 Pro Tips

### Improve Accuracy

1. **Download all images** - missing images = lower accuracy
2. **Use GPU** - faster iteration = more experiments
3. **Tune hyperparameters** - adjust XGBoost settings
4. **Add features** - extract more from text

### Save Time

1. **Start with text-only** - quick baseline
2. **Test on subset** - use 10K samples first
3. **Monitor training** - stop if validation doesn't improve

### Debug Effectively

1. **Run baseline first** - `python sample_baseline.py`
2. **Check each step** - verify data, features, model
3. **Use validation set** - never tune on test

---

## 📊 Performance Expectations

| Setup | Training SMAPE | Validation SMAPE | Time |
|-------|---------------|------------------|------|
| **With GPU + Images** | 0.15-0.18 | 0.20-0.25 | 1-2h |
| **With CPU + Images** | 0.15-0.18 | 0.20-0.25 | 3-4h |
| **Text-only** | 0.22-0.25 | 0.25-0.30 | 30m |
| **Baseline** | - | - | 1m |

---

## 🎯 Next Steps

After getting predictions:

1. ✅ Submit `test_out.csv`
2. ✅ Fill `Documentation.md` with team details
3. ✅ Submit documentation
4. ✅ (Optional) Try improvements for better score

---

## 📞 Need Help?

1. Read error messages carefully
2. Check this guide's troubleshooting section
3. Run `setup_check.py` to verify environment
4. Try text-only model as fallback

---

**You're all set! Start with Step 1 and follow through. Good luck! 🎉**
