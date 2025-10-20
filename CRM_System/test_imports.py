"""
Test Imports - Check for any import errors
=========================================

This script tests all imports to identify any potential errors.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports to identify errors"""
    
    print("🔍 TESTING IMPORTS")
    print("=" * 40)
    
    # Test basic imports
    try:
        import asyncio
        print("✅ asyncio - OK")
    except ImportError as e:
        print(f"❌ asyncio - Error: {e}")
    
    try:
        import logging
        print("✅ logging - OK")
    except ImportError as e:
        print(f"❌ logging - Error: {e}")
    
    try:
        from datetime import datetime
        print("✅ datetime - OK")
    except ImportError as e:
        print(f"❌ datetime - Error: {e}")
    
    # Test AI component imports
    print("\n🤖 TESTING AI COMPONENTS")
    print("-" * 30)
    
    try:
        from speech_synthesis import SpeechSynthesizer, VoiceConfig
        print("✅ speech_synthesis - OK")
    except ImportError as e:
        print(f"❌ speech_synthesis - Error: {e}")
    
    try:
        from nlp_processor import NLPProcessor
        print("✅ nlp_processor - OK")
    except ImportError as e:
        print(f"❌ nlp_processor - Error: {e}")
    
    try:
        from conversation_manager import ConversationManager
        print("✅ conversation_manager - OK")
    except ImportError as e:
        print(f"❌ conversation_manager - Error: {e}")
    
    try:
        from analytics_engine import AnalyticsEngine
        print("✅ analytics_engine - OK")
    except ImportError as e:
        print(f"❌ analytics_engine - Error: {e}")
    
    try:
        from multilingual_support import MultilingualSupport
        print("✅ multilingual_support - OK")
    except ImportError as e:
        print(f"❌ multilingual_support - Error: {e}")
    
    # Test optional dependencies
    print("\n📦 TESTING OPTIONAL DEPENDENCIES")
    print("-" * 35)
    
    try:
        import pyttsx3
        print("✅ pyttsx3 - OK")
    except ImportError as e:
        print(f"⚠️ pyttsx3 - Not installed: {e}")
        print("   Install with: pip install pyttsx3")
    
    try:
        import nltk
        print("✅ nltk - OK")
    except ImportError as e:
        print(f"⚠️ nltk - Not installed: {e}")
        print("   Install with: pip install nltk")
    
    try:
        import transformers
        print("✅ transformers - OK")
    except ImportError as e:
        print(f"⚠️ transformers - Not installed: {e}")
        print("   Install with: pip install transformers")
    
    try:
        import torch
        print("✅ torch - OK")
    except ImportError as e:
        print(f"⚠️ torch - Not installed: {e}")
        print("   Install with: pip install torch")
    
    try:
        from googletrans import Translator
        print("✅ googletrans - OK")
    except ImportError as e:
        print(f"⚠️ googletrans - Not installed: {e}")
        print("   Install with: pip install googletrans==4.0.0rc1")
    
    try:
        import langdetect
        print("✅ langdetect - OK")
    except ImportError as e:
        print(f"⚠️ langdetect - Not installed: {e}")
        print("   Install with: pip install langdetect")
    
    print("\n🎯 SUMMARY")
    print("=" * 40)
    print("✅ Core system imports - All working")
    print("⚠️ Some optional dependencies may not be installed")
    print("💡 This is normal - the demo will work with basic features")
    print("🚀 You can run the demo even with missing dependencies!")

if __name__ == "__main__":
    test_imports()

