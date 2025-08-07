# News RAG Summarizer

A comprehensive news aggregation and summarization tool that uses Retrieval-Augmented Generation (RAG) with LangChain to provide intelligent news summaries from multiple RSS sources.

## Features

- 🤖 **Multiple LLM Support**: OpenAI, Ollama (local), and LMStudio (local)
- 📰 **RSS Feed Integration**: BBC, CNN, Reuters, AP News, NPR, Al Jazeera, The Guardian, and more
- 🎯 **Customizable Areas of Interest**: Focus on up to 10 topics that matter to you
- 🔍 **RAG-powered Summarization**: Uses vector similarity search for context-aware summaries
- 📦 **Download-friendly Output**: Creates zip packages with Markdown and JSON outputs
- 🛠️ **Easy Configuration**: JSON-based configuration with interactive setup

## Quick Start

### 1. Download and Extract
Download the ZIP package and extract all files to a directory.

### 2. Setup
```bash
python setup.py
```

This will:
- Install required dependencies
- Set up configuration interactively
- Create run scripts

### 3. Configure (if needed)
Edit `config.json` to customize:
- Model provider and settings
- Areas of interest
- RSS feeds
- Output preferences

### 4. Run
```bash
python news_rag_summarizer.py
# or
python run.py
```

## Configuration

### Model Providers

#### OpenAI
```json
{
  "model_provider": "openai",
  "openai": {
    "api_key": "your-api-key-here",
    "model": "gpt-3.5-turbo",
    "temperature": 0.1
  }
}
```

#### Ollama (Local)
```json
{
  "model_provider": "ollama",
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "llama2",
    "temperature": 0.1
  }
}
```

#### LMStudio (Local)
```json
{
  "model_provider": "lmstudio",
  "lmstudio": {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "local-model",
    "temperature": 0.1
  }
}
```

### Areas of Interest
Customize your focus areas (up to 10):
```json
{
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
  ]
}
```

### RSS Feeds
The application includes these default RSS feeds:
- **BBC**: World news and updates
- **CNN**: Breaking news and analysis
- **Reuters**: International news
- **AP News**: Associated Press news
- **NPR**: National Public Radio
- **Al Jazeera**: Middle East and international news
- **The Guardian**: UK and world news

You can add more feeds in the configuration:
```json
{
  "rss_feeds": {
    "Custom Source": "https://example.com/rss"
  }
}
```

## Requirements

- Python 3.8+
- Internet connection for RSS feeds
- API key (for OpenAI) or local LLM setup (Ollama/LMStudio)

### Dependencies
- langchain
- openai
- requests
- feedparser
- faiss-cpu
- sentence-transformers
- beautifulsoup4

## Local LLM Setup

### Ollama
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Download a model: `ollama pull llama2`
3. Set configuration to use Ollama
4. Install Ollama support: `pip install langchain-ollama`

### LMStudio
1. Download LMStudio from [lmstudio.ai](https://lmstudio.ai)
2. Download and load a model
3. Start the local server
4. Set configuration to use LMStudio

## Output

The application generates:

1. **Markdown Summary** (`news_summary_YYYYMMDD_HHMMSS.md`)
   - Human-readable summary
   - Areas of interest covered
   - Source information

2. **JSON Data** (`news_data_YYYYMMDD_HHMMSS.json`)
   - Structured summary
   - Complete article data
   - Metadata

3. **ZIP Package** (`news_summary_package_YYYYMMDD_HHMMSS.zip`)
   - Contains all outputs
   - Configuration used
   - README file

## Usage Examples

### Basic Usage
```bash
python news_rag_summarizer.py
```

### Custom Configuration
1. Edit `config.json`
2. Modify areas of interest
3. Add/remove RSS feeds
4. Run the application

### Automated Scheduling
Create a cron job (Linux/Mac) or scheduled task (Windows):
```bash
# Daily at 8 AM
0 8 * * * cd /path/to/news-rag && python news_rag_summarizer.py
```

## Troubleshooting

### Common Issues

**"No articles collected"**
- Check internet connection
- Verify RSS feed URLs are accessible
- Check log file for detailed errors

**API Key Errors**
- Ensure OpenAI API key is set in config.json or environment variable
- Verify API key is valid and has sufficient credits

**Local LLM Connection Issues**
- Ensure Ollama/LMStudio is running
- Check base URL in configuration
- Verify model is loaded and accessible

**Memory Issues**
- Reduce `max_articles_per_source` in config
- Use smaller embedding models
- Consider using CPU instead of GPU for embeddings

### Log Files
Check `news_rag.log` for detailed error information and application flow.

## Customization

### Adding New RSS Feeds
Edit `config.json` and add feeds to the `rss_feeds` section:
```json
{
  "rss_feeds": {
    "New Source": "https://newssite.com/rss"
  }
}
```

### Custom Prompts
Modify the prompt template in `RAGProcessor.create_summary_chain()` method.

### Different Embeddings
Change the embeddings model in configuration:
```json
{
  "embeddings": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-mpnet-base-v2"
  }
}
```

## Advanced Usage

### Environment Variables
Set environment variables instead of config file:
```bash
export OPENAI_API_KEY="your-key-here"
export NEWS_RAG_CONFIG="/path/to/config.json"
```

### Batch Processing
Run multiple configurations:
```bash
for config in config*.json; do
    NEWS_RAG_CONFIG=$config python news_rag_summarizer.py
done
```

## License

This project is open source. Feel free to modify and distribute.

## Support

For issues and questions:
1. Check the log files
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify your configuration is valid JSON

## Contributing

To contribute:
1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

---

**Note**: This tool is for personal and educational use. Please respect the terms of service of RSS feed providers and LLM services.
