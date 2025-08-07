#!/usr/bin/env python3
"""
Example usage and configuration examples for News RAG Summarizer
"""

# Example 1: Using with OpenAI
OPENAI_CONFIG = {
    "model_provider": "openai",
    "openai": {
        "api_key": "sk-your-openai-api-key-here",
        "model": "gpt-3.5-turbo",
        "temperature": 0.1
    },
    "embeddings": {
        "provider": "openai"
    },
    "areas_of_interest": [
        "artificial intelligence",
        "climate change", 
        "cybersecurity",
        "renewable energy",
        "space exploration"
    ]
}

# Example 2: Using with Ollama (local)
OLLAMA_CONFIG = {
    "model_provider": "ollama",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "temperature": 0.1
    },
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "areas_of_interest": [
        "technology",
        "healthcare innovations",
        "economic trends",
        "environmental news",
        "scientific breakthroughs"
    ]
}

# Example 3: Using with LMStudio (local)
LMSTUDIO_CONFIG = {
    "model_provider": "lmstudio",
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "local-model",
        "temperature": 0.1
    },
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-mpnet-base-v2"
    },
    "areas_of_interest": [
        "machine learning",
        "blockchain",
        "quantum computing",
        "biotechnology",
        "renewable energy"
    ]
}

# Example 4: Custom RSS feeds
CUSTOM_RSS_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    "Nature News": "http://feeds.nature.com/nature/rss/current",
    "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
    "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Wired": "https://www.wired.com/feed/"
}

def create_custom_config():
    """Create a custom configuration file"""
    import json
    
    custom_config = {
        "model_provider": "openai",  # Change as needed
        "openai": {
            "api_key": "",  # Set your API key
            "model": "gpt-4",  # or gpt-3.5-turbo
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
            "artificial intelligence",
            "climate change",
            "cybersecurity", 
            "space exploration",
            "renewable energy",
            "biotechnology",
            "quantum computing",
            "autonomous vehicles",
            "digital privacy",
            "sustainable technology"
        ],
        "rss_feeds": {
            # Default news sources
            "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
            "CNN": "http://rss.cnn.com/rss/edition.rss",
            "Reuters": "https://feeds.reuters.com/reuters/topNews",
            "AP News": "https://feeds.apnews.com/apnews/topnews",
            "NPR": "https://feeds.npr.org/1001/rss.xml",
            
            # Tech-focused sources
            "TechCrunch": "https://techcrunch.com/feed/",
            "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            
            # Science sources
            "Nature News": "http://feeds.nature.com/nature/rss/current",
            "Science Daily": "https://www.sciencedaily.com/rss/all.xml"
        },
        "output": {
            "max_articles_per_source": 15,
            "summary_length": "detailed",
            "include_sources": True,
            "output_format": "both"
        }
    }
    
    # Save to file
    with open("custom_config.json", "w") as f:
        json.dump(custom_config, f, indent=2)
    
    print("✅ Custom configuration saved to: custom_config.json")
    print("💡 Edit the file and rename to config.json to use")

def run_example():
    """Run the application with example configuration"""
    import subprocess
    import sys
    import os
    
    # Create example config if it doesn't exist
    if not os.path.exists("config.json"):
        create_custom_config()
        print("\n📝 Please edit custom_config.json with your settings")
        print("🔄 Then rename it to config.json and run again")
        return
    
    # Run the main application
    print("🚀 Running News RAG Summarizer...")
    try:
        subprocess.run([sys.executable, "news_rag_summarizer.py"])
    except FileNotFoundError:
        print("❌ Main application not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("📋 News RAG Summarizer - Configuration Examples")
    print("=" * 50)
    
    print("\n1. Create custom configuration")
    print("2. Run with current configuration")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == "1":
        create_custom_config()
    elif choice == "2":
        run_example()
    else:
        print("Invalid choice")
