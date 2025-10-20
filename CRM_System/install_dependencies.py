"""
Install Dependencies for Real AI Demo
====================================

This script installs the required dependencies for the real AI demo.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all required dependencies"""
    
    print("🔧 INSTALLING DEPENDENCIES FOR REAL AI DEMO")
    print("=" * 50)
    
    # Required packages for real AI demo
    packages = [
        "pyttsx3>=2.90",
        "nltk>=3.8.1",
        "textblob>=0.17.1",
        "vaderSentiment>=3.3.2",
        "googletrans==4.0.0rc1",
        "langdetect>=1.0.9",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0"
    ]
    
    print("Installing packages...")
    print()
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
            success_count += 1
        else:
            print(f"❌ Failed to install {package}")
        print()
    
    print("=" * 50)
    print(f"Installation complete: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        print("You can now run the real AI demo!")
    else:
        print("⚠️ Some packages failed to install.")
        print("You can still run the demo, but some features might not work.")
    
    print()
    print("To run the real AI demo:")
    print("  python run_real_demo.py")
    print("  or double-click run_real_demo.bat")

if __name__ == "__main__":
    main()
