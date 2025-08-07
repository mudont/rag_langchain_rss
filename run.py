#!/usr/bin/env python3
import subprocess
import sys

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "news_rag_summarizer.py"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
