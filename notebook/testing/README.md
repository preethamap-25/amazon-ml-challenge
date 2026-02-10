# ML Challenge 2025 - Product Price Prediction

Complete ML solution for predicting product prices using images and text features.

## 📋 Overview

This solution uses a **multimodal approach** combining:
- **Image features**: Extracted using pre-trained ResNet50 CNN
- **Text features**: Extracted from catalog content (IPQ, keywords, specs)
- **Ensemble model**: XGBoost for final price prediction

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Place your dataset files in the `dataset/` folder:
```
dataset/
├── train.csv
├── test.csv
└── sample_test.csv
```

### Step 3: Download Images

Run the image download script to fetch all product images:

```bash
python download_images.py
```

This will download images from the URLs in your CSV files into the `images/` folder. The script includes:
- ✓ Automatic retry logic for failed downloads
- ✓ Progress tracking
- ✓ Skip already downloaded images

**Note**: This may take 30-60 minutes depending on your internet speed and the number of images.

### Step 4: Train Model and Generate Predictions

Run the main pipeline:

```bash
python train_and_predict.py
```

This script will:
1. ✓ Load training data
2. ✓ Extract image features using ResNet50
3. ✓ Extract text features from catalog content
4. ✓ Train XGBoost model
5. ✓ Generate predictions on test data
6. ✓ Save output to `test_out.csv`

**Expected runtime**: 1-2 hours depending on your hardware (GPU recommended)

### Step 5: Submit Results

Your predictions will be saved in `test_out.csv` with the format:
```csv
sample_id,price
1,29.99
2,149.99
...
```

Upload this file to the competition portal.

## 📁 Project Structure

```
ml_challenge_2025/
├── train_and_predict.py      # Main pipeline script
├── download_images.py         # Image download utility
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── dataset/                   # Your CSV files
│   ├── train.csv
│   └── test.csv
├── images/                    # Downloaded product images
├── models/                    # Saved model files
│   └── price_predictor.pkl
└── test_out.csv              # Final predictions (output)
```

## 🔧 How It Works

### 1. Image Feature Extraction

- Uses **ResNet50** pre-trained on ImageNet
- Extracts 2048-dimensional feature vectors
- Images are preprocessed to 224×224 pixels
- Features capture visual characteristics that correlate with price

### 2. Text Feature Extraction

From the catalog content, we extract:
- **Text length** and word count
- **Item Pack Quantity (IPQ)**: Number of items in pack
- **Brand detection**: Presence of known brands
- **Premium/budget keywords**: Quality indicators
- **Numeric specs**: Max and average numbers (GB, MP, etc.)

### 3. Model Training

- **Algorithm**: XGBoost Regressor
- **Why XGBoost?**: 
  - Best performance on tabular data
  - Handles feature interactions well
  - Fast training and prediction
- **Features**: Combined image (2048) + text (7) = 2055 total features
- **Validation**: 20% held-out set for monitoring performance

### 4. Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**:
```
SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
```

Lower is better (0 = perfect predictions)

## 🎯 Performance Tips

### Get Better Results

1. **Download all images**: Missing images = missing features
2. **Use GPU**: Add `--gpu` if you have CUDA available
3. **Hyperparameter tuning**: Adjust XGBoost parameters in config
4. **Feature engineering**: Add more text features from catalog

### System Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Storage**: ~5GB for images + models

## 🐛 Troubleshooting

### Images not downloading?

- Check internet connection
- Some URLs may be broken - this is normal
- The model handles missing images with zero vectors

### Out of memory error?

Reduce batch size in `train_and_predict.py`:
```python
Config.BATCH_SIZE = 16  # Change from 32 to 16
```

### CUDA out of memory?

Force CPU usage:
```python
Config.DEVICE = 'cpu'  # Change from 'cuda'
```

## 📊 Expected Results

With this solution, you should achieve:
- **Training SMAPE**: ~0.15-0.20
- **Validation SMAPE**: ~0.18-0.25
- **Test SMAPE**: ~0.20-0.30 (estimated)

## 🔬 Advanced Customization

### Use Different Image Model

Replace ResNet50 with EfficientNet in `train_and_predict.py`:

```python
from torchvision import models

# Option 1: EfficientNet-B0 (faster, smaller)
self.model = models.efficientnet_b0(pretrained=True)

# Option 2: EfficientNet-B3 (more accurate)
self.model = models.efficientnet_b3(pretrained=True)
```

### Add More Text Features

Enhance `TextFeatureExtractor.extract_numeric_features()`:

```python
# Add custom features
features['has_warranty'] = int('warranty' in text)
features['has_rating'] = int('star' in text or 'rating' in text)
# etc.
```

### Try Different Models

Replace XGBoost with alternatives:

```python
# Random Forest
from sklearn.ensemble import RandomForestRegressor
self.model = RandomForestRegressor(n_estimators=500, max_depth=20)

# LightGBM (faster than XGBoost)
import lightgbm as lgb
self.model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
```

## 📝 Submission Checklist

- [ ] Downloaded all images using `download_images.py`
- [ ] Trained model using `train_and_predict.py`
- [ ] Generated `test_out.csv` with predictions
- [ ] Verified output format matches `sample_test_out.csv`
- [ ] Checked all test sample_ids are present
- [ ] All predicted prices are positive
- [ ] Prepared 1-page methodology document

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Make sure dataset files are in the right location
4. Check that images are downloaded successfully

## 📄 License

MIT License - Free to use and modify

---

**Good luck with the challenge! 🎯**
