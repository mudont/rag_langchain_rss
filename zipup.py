#!/usr/bin/env python3
"""
Script to create a downloadable zip package with all necessary files
"""

import zipfile
import os
from datetime import datetime

def create_package():
    """Create a zip package with all application files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"news_rag_summarizer_{timestamp}.zip"
    
    files_to_include = [
        ("news_rag_summarizer.py", "Main application file"),
        ("setup.py", "Setup and installation script"),
        ("requirements.txt", "Python dependencies"),
        ("config.json", "Configuration file (template)"),
        ("README.md", "Documentation and usage guide"),
        ("run.py", "Simple run script"),
    ]
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, description in files_to_include:
            if os.path.exists(filename):
                zipf.write(filename)
                print(f"✅ Added: {filename} - {description}")
            else:
                print(f"⚠️  Missing: {filename}")
        
        # Create installation instructions
        install_instructions = """# Installation Instructions

## Quick Start
1. Extract all files to a directory
2. Run: python setup.py
3. Follow the interactive setup
4. Run: python news_rag_summarizer.py

## Manual Setup
1. Install dependencies: pip install -r requirements.txt
2. Copy config.json to your desired settings
3. Set your API keys in config.json
4. Run: python news_rag_summarizer.py

## Files Included
- news_rag_summarizer.py: Main application
- setup.py: Interactive setup script
- requirements.txt: Python dependencies
- config.json: Configuration template
- README.md: Complete documentation
- run.py: Simple run script

For detailed instructions, see README.md
"""
        
        zipf.writestr("INSTALL.txt", install_instructions)
        
        # Create a simple launcher script for Windows
        windows_launcher = """@echo off
echo News RAG Summarizer
echo ==================
python setup.py
pause
"""
        zipf.writestr("setup.bat", windows_launcher)
        
        # Create a Unix launcher script
        unix_launcher = """#!/bin/bash
echo "News RAG Summarizer"
echo "=================="
python3 setup.py
"""
        zipf.writestr("setup.sh", unix_launcher)
    
    print(f"\n🎉 Package created: {package_name}")
    print(f"📦 Package size: {os.path.getsize(package_name)} bytes")
    return package_name

if __name__ == "__main__":
    create_package()
