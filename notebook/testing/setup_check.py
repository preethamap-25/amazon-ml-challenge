"""
Setup Verification Script
Run this to verify your environment is ready for the ML challenge
"""

import sys
import importlib
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ⚠️  Warning: Python 3.7+ recommended")
        return False
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False


def check_directory_structure():
    """Check if required directories exist"""
    dirs = ['dataset', 'models', 'images']
    all_exist = True
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing - will be created")
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return all_exist


def check_dataset_files():
    """Check if dataset files exist"""
    files = ['dataset/train.csv', 'dataset/test.csv']
    all_exist = True
    
    for file_path in files:
        if os.path.exists(file_path):
            # Get file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("✗ No GPU detected - will use CPU")
            print("  Training will be slower but still works")
            return False
    except:
        print("✗ Cannot check GPU")
        return False


def main():
    """Main verification"""
    print("="*60)
    print("ML CHALLENGE 2025 - SETUP VERIFICATION")
    print("="*60)
    
    all_ok = True
    
    # Python version
    print("\n1. Python Version:")
    if not check_python_version():
        all_ok = False
    
    # Required packages
    print("\n2. Required Packages:")
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'PIL'),
        ('sklearn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('tqdm', 'tqdm'),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_ok = False
    
    # Directory structure
    print("\n3. Directory Structure:")
    check_directory_structure()
    
    # Dataset files
    print("\n4. Dataset Files:")
    if not check_dataset_files():
        print("\n  ⚠️  Please place your dataset files in the dataset/ folder:")
        print("     - dataset/train.csv")
        print("     - dataset/test.csv")
        all_ok = False
    
    # GPU
    print("\n5. GPU Check:")
    has_gpu = check_gpu()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\nTo install, run:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
    
    if all_ok:
        print("\n✓✓✓ All checks passed! ✓✓✓")
        print("\nYou're ready to run:")
        print("  1. python download_images.py  (download product images)")
        print("  2. python train_and_predict.py  (train model and predict)")
        print("\nOR for faster text-only approach:")
        print("  python train_text_only.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
    
    if not has_gpu:
        print("\n💡 TIP: GPU not available. Training will work but be slower.")
        print("   Consider using Google Colab with GPU for faster training.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
