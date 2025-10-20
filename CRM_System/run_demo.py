"""
Quick Demo Runner for Teacher
============================

Run this script to show your teacher how the AI calling system works.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Run the teacher demonstration"""
    
    print("🚀 Starting AI Calling System Demo...")
    print("This will show your teacher how the system works!")
    print()
    
    try:
        # Import and run the demo
        from teacher_demo import main as demo_main
        await demo_main()
        
    except ImportError as e:
        print(f"❌ Error importing demo: {e}")
        print("Make sure all files are in the same directory.")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 AI CALLING SYSTEM - TEACHER DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Run the demo
    asyncio.run(main())
