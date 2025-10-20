"""
Real AI Demo Runner
==================

Run this to show your teacher the ACTUAL AI system working!
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Run the real AI demonstration"""
    
    print("🎓 REAL AI CALLING SYSTEM - TEACHER DEMONSTRATION")
    print("=" * 60)
    print("This will show the ACTUAL AI system working with real:")
    print("✅ Speech synthesis (TTS)")
    print("✅ Natural language processing (NLP)")
    print("✅ Conversation management")
    print("✅ Analytics and insights")
    print("✅ Multilingual support")
    print("=" * 60)
    print()
    
    try:
        # Import and run the real demo
        from real_demo import RealAIDemo
        
        demo = RealAIDemo()
        await demo.run_complete_demo()
        
    except ImportError as e:
        print(f" Error importing AI components: {e}")
        print(" Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print()
        print("Required packages:")
        print("   - pyttsx3 (for speech synthesis)")
        print("   - nltk (for NLP processing)")
        print("   - transformers (for AI models)")
        print("   - googletrans (for translation)")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        print(" This might be due to missing dependencies")
        print("   Try: pip install -r requirements.txt")

if __name__ == "__main__":
    print(" Starting Real AI Demo...")
    print("This will show your teacher the ACTUAL AI system working!")
    print()
    
    # Run the real demo
    asyncio.run(main())
