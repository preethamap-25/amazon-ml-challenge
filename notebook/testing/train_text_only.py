"""
ALTERNATIVE SOLUTION: Text-Only Price Prediction
Use this if you have issues downloading images

This script uses ONLY text features from catalog_content
Faster to run, but slightly lower accuracy than multimodal approach
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


class Config:
    """Configuration"""
    TRAIN_CSV = 'dataset/train.csv'
    TEST_CSV = 'dataset/test.csv'
    OUTPUT_PATH = 'test_out.csv'
    MODEL_PATH = 'models/text_only_model.pkl'
    RANDOM_SEED = 42
    TEST_SIZE = 0.2


class TextFeatureExtractor:
    """Extract comprehensive features from text"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2
        )
        self.brand_keywords = [
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 
            'asus', 'acer', 'nike', 'adidas', 'puma', 'canon', 'nikon',
            'microsoft', 'google', 'amazon', 'philips', 'panasonic'
        ]
    
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
            r'count[:\s]*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                qty = int(match.group(1))
                if qty <= 100:  # Sanity check
                    return qty
        return 1
    
    def extract_manual_features(self, texts):
        """Extract hand-crafted features"""
        features_list = []
        
        for text in tqdm(texts, desc="Extracting manual features"):
            if pd.isna(text):
                text = ""
            
            text_lower = str(text).lower()
            
            features = {
                # Basic stats
                'text_length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
                
                # IPQ
                'ipq': self.extract_ipq(text),
                
                # Brand
                'has_brand': int(any(brand in text_lower for brand in self.brand_keywords)),
                
                # Quality indicators
                'has_premium': int(any(word in text_lower for word in 
                                      ['premium', 'luxury', 'pro', 'ultra', 'deluxe', 'professional'])),
                'has_budget': int(any(word in text_lower for word in 
                                     ['budget', 'basic', 'economy', 'value', 'affordable'])),
                
                # Product categories (inferred)
                'is_electronics': int(any(word in text_lower for word in 
                                         ['electronic', 'digital', 'wireless', 'bluetooth', 'usb', 'hdmi'])),
                'is_clothing': int(any(word in text_lower for word in 
                                      ['shirt', 'pant', 'dress', 'shoe', 'clothing', 'apparel'])),
                'is_food': int(any(word in text_lower for word in 
                                  ['food', 'snack', 'beverage', 'drink', 'organic'])),
                
                # Numeric features
                'number_count': len(re.findall(r'\d+', text)),
            }
            
            # Extract numbers (specs like GB, MP, etc.)
            numbers = re.findall(r'\d+', text)
            if numbers:
                numbers = [int(n) for n in numbers if int(n) < 10000]  # Filter outliers
                if numbers:
                    features['max_number'] = max(numbers)
                    features['avg_number'] = np.mean(numbers)
                    features['min_number'] = min(numbers)
                else:
                    features['max_number'] = 0
                    features['avg_number'] = 0
                    features['min_number'] = 0
            else:
                features['max_number'] = 0
                features['avg_number'] = 0
                features['min_number'] = 0
            
            # Special characters
            features['exclamation_count'] = text.count('!')
            features['question_count'] = text.count('?')
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def fit_transform(self, texts):
        """Fit and transform texts to features"""
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf.fit_transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        manual_features = self.extract_manual_features(texts)
        
        combined = pd.concat([manual_features, tfidf_df], axis=1)
        print(f"Total features: {combined.shape[1]}")
        return combined
    
    def transform(self, texts):
        """Transform texts to features (inference)"""
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf.transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        manual_features = self.extract_manual_features(texts)
        
        combined = pd.concat([manual_features, tfidf_df], axis=1)
        return combined


def calculate_smape(y_true, y_pred):
    """Calculate SMAPE"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    return np.mean(smape)


def main():
    print("="*60)
    print("TEXT-ONLY PRICE PREDICTION")
    print("="*60)
    
    # Set seed
    np.random.seed(Config.RANDOM_SEED)
    
    # Load data
    print(f"\nLoading training data from {Config.TRAIN_CSV}...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"Loaded {len(train_df)} samples")
    
    # Extract features
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    extractor = TextFeatureExtractor()
    X = extractor.fit_transform(train_df['catalog_content'].fillna(''))
    y = train_df['price'].values
    
    # Split data
    print(f"\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_smape = calculate_smape(y_train, train_pred)
    val_smape = calculate_smape(y_val, val_pred)
    
    print(f"\n{'='*60}")
    print(f"Training SMAPE: {train_smape:.4f}")
    print(f"Validation SMAPE: {val_smape:.4f}")
    print(f"{'='*60}")
    
    # Save model
    import os
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    with open(Config.MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'extractor': extractor
        }, f)
    print(f"\nModel saved to {Config.MODEL_PATH}")
    
    # Predict on test
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    print(f"\nLoading test data from {Config.TEST_CSV}...")
    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Loaded {len(test_df)} samples")
    
    print("\nExtracting features...")
    X_test = extractor.transform(test_df['catalog_content'].fillna(''))
    X_test_scaled = scaler.transform(X_test)
    
    print("Generating predictions...")
    predictions = model.predict(X_test_scaled)
    predictions = np.clip(predictions, 0.01, None)
    
    # Save predictions
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    output_df.to_csv(Config.OUTPUT_PATH, index=False)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Predictions saved to: {Config.OUTPUT_PATH}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Mean price: ${predictions.mean():.2f}")


if __name__ == "__main__":
    main()
