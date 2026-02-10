"""
Download Images from URLs
Run this script BEFORE training to download all product images
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
import os


def download_images(csv_path, image_dir='images', max_retries=3):
    """
    Download images from URLs in CSV file
    
    Args:
        csv_path: Path to CSV file with image_link column
        image_dir: Directory to save images
        max_retries: Number of retry attempts for failed downloads
    """
    # Create image directory
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples")
    
    # Download images
    print(f"\nDownloading images to {image_dir}/")
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        sample_id = row['sample_id']
        image_url = row['image_link']
        
        # Determine file extension
        file_ext = '.jpg'
        if isinstance(image_url, str) and '.' in image_url:
            file_ext = '.' + image_url.split('.')[-1].split('?')[0]
            if file_ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                file_ext = '.jpg'
        
        filename = f"{sample_id}{file_ext}"
        filepath = os.path.join(image_dir, filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            skipped_count += 1
            continue
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                if pd.isna(image_url):
                    failed_count += 1
                    break
                
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                success_count += 1
                time.sleep(0.1)  # Small delay to avoid rate limiting
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    failed_count += 1
                    print(f"\nFailed to download {sample_id}: {str(e)}")
                else:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Already existed (skipped): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {(success_count + skipped_count) / len(df) * 100:.1f}%")


if __name__ == "__main__":
    print("="*60)
    print("IMAGE DOWNLOAD SCRIPT")
    print("="*60)
    
    # Download training images
    print("\n[1/2] Downloading TRAINING images...")
    download_images('dataset/train.csv', 'images')
    
    # Download test images
    print("\n[2/2] Downloading TEST images...")
    download_images('dataset/test.csv', 'images')
    
    print("\n✓ Image download complete!")
    print("\nYou can now run: python train_and_predict.py")
