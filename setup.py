#!/usr/bin/env python3
"""
Setup script for News RAG Summarizer
"""

import os
import sys
import json
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_config():
    """Set up initial configuration"""
    config_file = "config.json"
    
    if os.path.exists(config_file):
        print(f"⚠️  Configuration file '{config_file}' already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("📋 Using existing configuration")
            return True
    
    print("🔧 Setting up configuration...")
    
    # Get user preferences
    print("\n1. Model Provider Selection:")
    print("   1) OpenAI (requires API key)")
    print("   2) Ollama (local)")
    print("   3) LMStudio (local)")
    
    while True:
        choice = input("Choose model provider (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    provider_map = {'1': 'openai', '2': 'ollama', '3': 'lmstudio'}
    provider = provider_map[choice]
    
    # Load default config
    with open(config_file + '.template', 'w') as f:
        f.write("""
{
  "model_provider": "openai",
  "openai": {
    "api_key": "",
    "model": "gpt-3.5-turbo",
    "temperature": 0.1
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "llama2",
    "temperature": 0.1
  },
  "lmstudio": {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "local-model",
    "temperature": 0.1
  },
  "embeddings": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "areas_of_interest": [
    "technology",
    "politics", 
    "business",
    "health",
    "science",
    "sports",
    "entertainment",
    "world news",
    "climate",
    "economy"
  ],
  "rss_feeds": {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "AP News": "https://feeds.apnews.com/apnews/topnews",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "Associated Press": "https://feeds.apnews.com/apnews/world"
  },
  "output": {
    "max_articles_per_source": 20,
    "summary_length": "medium",
    "include_sources": true,
    "output_format": "both"
  }
}
""")
    
    with open(config_file + '.template', 'r') as f:
        config = json.load(f)
    
    os.remove(config_file + '.template')
    
    # Update provider
    config['model_provider'] = provider
    
    # Get API key for OpenAI if selected
    if provider == 'openai':
        api_key = input("Enter your OpenAI API key (or press Enter to set later): ").strip()
        if api_key:
            config['openai']['api_key'] = api_key
        else:
            print("💡 You can set the API key later in config.json or as OPENAI_API_KEY environment variable")
    
    # Customize areas of interest
    print(f"\n2. Current areas of interest: {', '.join(config['areas_of_interest'])}")
    customize = input("Do you want to customize areas of interest? (y/N): ").lower()
    
    if customize == 'y':
        print("Enter up to 10 areas of interest (press Enter when done):")
        new_areas = []
        for i in range(10):
            area = input(f"Area {i+1}: ").strip()
            if not area:
                break
            new_areas.append(area)
        
        if new_areas:
            config['areas_of_interest'] = new_areas
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_file}")
    return True

def create_run_script():
    """Create a simple run script"""
    script_content = """#!/usr/bin/env python3
import subprocess
import sys

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "news_rag_summarizer.py"])
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
"""
    
    with open("run.py", "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    try:
        os.chmod("run.py", 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    print("✅ Run script created: run.py")

def main():
    """Main setup function"""
    print("🚀 News RAG Summarizer Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Setup configuration
    if not setup_config():
        return 1
    
    # Create run script
    create_run_script()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit config.json if needed")
    print("2. Run: python news_rag_summarizer.py")
    print("   or: python run.py")
    
    if input("\nRun the application now? (y/N): ").lower() == 'y':
        print("\n" + "="*50)
        import news_rag_summarizer
        return news_rag_summarizer.main()
    
    return 0

if __name__ == "__main__":
    exit(main())
