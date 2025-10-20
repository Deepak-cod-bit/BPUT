"""
Safe Demo Runner - Works Even With Missing Dependencies
======================================================

This demo will work even if some AI dependencies are missing.
It shows the system working with whatever components are available.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Run the safe AI demonstration"""
    
    print("🎓 SAFE AI CALLING SYSTEM - TEACHER DEMONSTRATION")
    print("=" * 60)
    print("This demo will work even if some dependencies are missing!")
    print("It shows the AI system working with available components.")
    print("=" * 60)
    print()
    
    try:
        # Import and run the safe demo
        from safe_demo import SafeAIDemo
        
        demo = SafeAIDemo()
        await demo.run_complete_demo()
        
    except ImportError as e:
        print(f"❌ Error importing demo: {e}")
        print("💡 Make sure the safe_demo.py file is in the same directory")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("💡 This is a safe demo - it should work even with missing dependencies")

if __name__ == "__main__":
    print("🚀 Starting Safe AI Demo...")
    print("This will work even if some dependencies are missing!")
    print()
    
    # Run the safe demo
    asyncio.run(main())

