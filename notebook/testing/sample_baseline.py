"""
Sample Baseline Code - Price Prediction
This generates a simple baseline using basic statistics

Use this to:
1. Understand the output format
2. Get a quick baseline score
3. Test your submission pipeline
"""

import pandas as pd
import numpy as np

def create_baseline_predictions(test_csv_path='dataset/test.csv', 
                                output_path='baseline_out.csv',
                                train_csv_path='dataset/train.csv'):
    """
    Create baseline predictions using simple statistics from training data
    
    Strategy: Use mean price from training data with small random variation
    """
    
    print("="*60)
    print("BASELINE PRICE PREDICTION")
    print("="*60)
    
    # Load training data to get price statistics
    print(f"\nLoading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    # Calculate price statistics
    mean_price = train_df['price'].mean()
    std_price = train_df['price'].std()
    min_price = train_df['price'].min()
    max_price = train_df['price'].max()
    median_price = train_df['price'].median()
    
    print(f"\nTraining Set Price Statistics:")
    print(f"  Mean: ${mean_price:.2f}")
    print(f"  Median: ${median_price:.2f}")
    print(f"  Std Dev: ${std_price:.2f}")
    print(f"  Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Load test data
    print(f"\nLoading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    print(f"Test samples: {len(test_df)}")
    
    # Generate baseline predictions
    # Strategy: Mean price + random noise
    np.random.seed(42)
    predictions = np.random.normal(
        loc=mean_price,
        scale=std_price * 0.5,  # Reduced variance
        size=len(test_df)
    )
    
    # Ensure positive prices and reasonable range
    predictions = np.clip(predictions, min_price * 0.5, max_price * 1.5)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("BASELINE PREDICTIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"Total predictions: {len(output_df)}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: ${predictions.mean():.2f}")
    print(f"  Median: ${np.median(predictions):.2f}")
    print(f"  Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    
    print(f"\n✓ Baseline file created successfully!")
    print(f"\nExpected SMAPE: ~0.40-0.50 (baseline)")
    print("This is your starting point - ML models should beat this!")
    
    return output_df


if __name__ == "__main__":
    # Generate baseline predictions
    create_baseline_predictions()
    
    # Show sample output
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (first 10 rows):")
    print("="*60)
    df = pd.read_csv('baseline_out.csv')
    print(df.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("OUTPUT FORMAT VALIDATION")
    print("="*60)
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Row count: {len(df)}")
    print(f"✓ All prices positive: {(df['price'] > 0).all()}")
    print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
