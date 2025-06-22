#!/usr/bin/env python3
"""
Setup script for SDG Multi-Label Classification Project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {platform.python_version()}")
        return False
    else:
        print(f"✅ Python version: {platform.python_version()}")
        return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install some packages!")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except ImportError:
        print("❌ NLTK not installed!")
        return False
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created: {directory}/")
        else:
            print(f"   📂 Exists: {directory}/")
    
    return True

def verify_data():
    """Verify that the data file exists"""
    print("\n🔍 Verifying data file...")
    
    data_path = "data/2503_to_3336_preprocessing_labeling.csv"
    
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # Size in MB
        print(f"   ✅ Data file found: {data_path} ({file_size:.1f} MB)")
        return True
    else:
        print(f"   ❌ Data file not found: {data_path}")
        print("   Please ensure the data file is in the correct location.")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'nltk',
        'joblib',
        'scipy'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def main():
    """Main setup function"""
    print("🚀 SDG Multi-Label Classification Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️ Some packages failed to install. You may need to install them manually.")
    
    # Test imports
    if not test_imports():
        print("\n⚠️ Some packages are not available. Please check the installation.")
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    create_directories()
    
    # Verify data
    data_ok = verify_data()
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. If data verification failed, ensure the CSV file is in data/ directory")
    print("2. Run 'python train_model.py' to train models")
    print("3. Or open 'notebooks/eda_visualization.ipynb' for interactive analysis")
    print("4. Use 'python predict_sample.py' for predictions after training")
    
    if not data_ok:
        print("\n⚠️ Warning: Data file not found. Training will fail without proper data.")
    
    print("\n🔗 For more information, see README_PROJECT.md")

if __name__ == "__main__":
    main()
