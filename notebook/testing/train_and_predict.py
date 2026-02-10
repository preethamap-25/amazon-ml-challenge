"""
ML Challenge 2025 - Product Price Prediction
Complete Step-by-Step Pipeline

This script handles:
1. Data loading
2. Image feature extraction using pre-trained CNN
3. Text feature extraction from catalog content
4. Model training with combined features
5. Price prediction on test data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters"""
    # Paths
    TRAIN_CSV = 'dataset/train.csv'
    TEST_CSV = 'dataset/test.csv'
    IMAGE_DIR = 'images'
    OUTPUT_PATH = 'test_out.csv'
    MODEL_SAVE_PATH = 'models/price_predictor.pkl'
    
    # Image processing
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    
    # Model
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# STEP 2: IMAGE FEATURE EXTRACTION
# ============================================================================

class ImageFeatureExtractor:
    """Extract features from product images using pre-trained ResNet"""
    
    def __init__(self, device='cpu'):
        print("Loading pre-trained ResNet50 model...")
        self.device = device
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {device}")
    
    def extract_features(self, image_paths):
        """Extract features from list of image paths"""
        features_list = []
        
        print(f"Extracting features from {len(image_paths)} images...")
        
        with torch.no_grad():
            for img_path in tqdm(image_paths):
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    features = self.model(img_tensor)
                    features = features.squeeze().cpu().numpy()
                    features_list.append(features)
                    
                except Exception as e:
                    # If image fails, use zero vector
                    print(f"Error processing {img_path}: {e}")
                    features_list.append(np.zeros(2048))
        
        return np.array(features_list)


# ============================================================================
# STEP 3: TEXT FEATURE EXTRACTION
# ============================================================================

class TextFeatureExtractor:
    """Extract features from catalog text"""
    
    def __init__(self):
        self.brand_keywords = ['apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 
                               'lenovo', 'asus', 'acer', 'nike', 'adidas', 
                               'puma', 'canon', 'nikon', 'microsoft']
    
    def extract_ipq(self, text):
        """Extract Item Pack Quantity"""
        if pd.isna(text):
            return 1
        
        patterns = [
            r'ipq[:\s]*(\d+)',
            r'pack of (\d+)',
            r'(\d+)-pack',
            r'(\d+)\s*pack',
            r'quantity[:\s]*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return int(match.group(1))
        return 1
    
    def extract_numeric_features(self, text):
        """Extract numeric features from text"""
        if pd.isna(text):
            text = ""
        
        text = str(text).lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_brand': int(any(brand in text for brand in self.brand_keywords)),
            'number_count': len(re.findall(r'\d+', text)),
            'ipq': self.extract_ipq(text),
        }
        
        # Price-related keywords
        features['has_premium'] = int(any(word in text for word in ['premium', 'luxury', 'pro', 'ultra']))
        features['has_budget'] = int(any(word in text for word in ['budget', 'basic', 'economy']))
        
        # Extract numbers (could be specs like GB, MP, etc.)
        numbers = re.findall(r'\d+', text)
        if numbers:
            features['max_number'] = max([int(n) for n in numbers])
            features['avg_number'] = np.mean([int(n) for n in numbers])
        else:
            features['max_number'] = 0
            features['avg_number'] = 0
        
        return features
    
    def extract_features(self, texts):
        """Extract features from list of texts"""
        print(f"Extracting text features from {len(texts)} samples...")
        
        all_features = []
        for text in tqdm(texts):
            features = self.extract_numeric_features(text)
            all_features.append(features)
        
        return pd.DataFrame(all_features)


# ============================================================================
# STEP 4: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(csv_path, image_dir, is_train=True):
    """Load data and prepare features"""
    print(f"\n{'='*60}")
    print(f"Loading data from {csv_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Create image paths
    df['image_path'] = df['sample_id'].apply(
        lambda x: f"{image_dir}/{x}.jpg"
    )
    
    return df


# ============================================================================
# STEP 5: FEATURE COMBINATION AND MODEL TRAINING
# ============================================================================

class PricePredictionModel:
    """Combined model for price prediction"""
    
    def __init__(self):
        self.image_extractor = None
        self.text_extractor = TextFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
    
    def extract_all_features(self, df, use_images=True):
        """Extract and combine all features"""
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60)
        
        # Extract text features
        text_features = self.text_extractor.extract_features(
            df['catalog_content'].values
        )
        
        if use_images:
            # Extract image features
            if self.image_extractor is None:
                self.image_extractor = ImageFeatureExtractor(
                    device=Config.DEVICE
                )
            
            image_features = self.image_extractor.extract_features(
                df['image_path'].values
            )
            
            # Create column names for image features
            img_cols = [f'img_feat_{i}' for i in range(image_features.shape[1])]
            image_df = pd.DataFrame(image_features, columns=img_cols)
            
            # Combine features
            combined_features = pd.concat([
                text_features.reset_index(drop=True),
                image_df.reset_index(drop=True)
            ], axis=1)
        else:
            combined_features = text_features
        
        print(f"\nTotal features: {combined_features.shape[1]}")
        return combined_features
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the price prediction model"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train XGBoost model (best for tabular data)
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=50
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_smape = self.calculate_smape(y_train, train_pred)
        val_smape = self.calculate_smape(y_val, val_pred)
        
        print(f"\nTraining SMAPE: {train_smape:.4f}")
        print(f"Validation SMAPE: {val_smape:.4f}")
        
        return val_smape
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Ensure positive prices
        predictions = np.clip(predictions, 0.01, None)
        return predictions
    
    @staticmethod
    def calculate_smape(y_true, y_pred):
        """Calculate SMAPE metric"""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        smape = np.zeros_like(numerator)
        smape[mask] = numerator[mask] / denominator[mask]
        return np.mean(smape)
    
    def save(self, path):
        """Save model"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'text_extractor': self.text_extractor
            }, f)
        print(f"\nModel saved to {path}")
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.text_extractor = data['text_extractor']
        print(f"\nModel loaded from {path}")


# ============================================================================
# STEP 6: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*60)
    print("ML CHALLENGE 2025 - PRICE PREDICTION PIPELINE")
    print("="*60)
    
    # Set random seeds
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    
    # ========================================================================
    # LOAD TRAINING DATA
    # ========================================================================
    train_df = load_and_prepare_data(
        Config.TRAIN_CSV, 
        Config.IMAGE_DIR, 
        is_train=True
    )
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    predictor = PricePredictionModel()
    
    # ========================================================================
    # EXTRACT FEATURES
    # ========================================================================
    print("\nExtracting features from training data...")
    X = predictor.extract_all_features(train_df, use_images=True)
    y = train_df['price'].values
    
    # ========================================================================
    # SPLIT DATA
    # ========================================================================
    print(f"\nSplitting data: {1-Config.TEST_SIZE:.0%} train, {Config.TEST_SIZE:.0%} validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_SEED
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    predictor.train(X_train, y_train, X_val, y_val)
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    predictor.save(Config.MODEL_SAVE_PATH)
    
    # ========================================================================
    # LOAD TEST DATA AND PREDICT
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS ON TEST DATA")
    print("="*60)
    
    test_df = load_and_prepare_data(
        Config.TEST_CSV, 
        Config.IMAGE_DIR, 
        is_train=False
    )
    
    # Extract features
    print("\nExtracting features from test data...")
    X_test = predictor.extract_all_features(test_df, use_images=True)
    
    # Predict
    print("\nMaking predictions...")
    predictions = predictor.predict(X_test)
    
    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    output_df.to_csv(Config.OUTPUT_PATH, index=False)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)
    print(f"Output saved to: {Config.OUTPUT_PATH}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Mean price: ${predictions.mean():.2f}")
    print(f"Median price: ${np.median(predictions):.2f}")
    
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
