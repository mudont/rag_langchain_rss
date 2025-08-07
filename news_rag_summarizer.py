#!/usr/bin/env python3
"""
News RAG Summarizer
A comprehensive news aggregation and summarization tool using RAG with LangChain.
Supports OpenAI, Ollama, and LMStudio models.
"""

import os
import json
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import time
import hashlib

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# For local models (Ollama/LMStudio)
try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
except ImportError:
    print("Ollama not available - install with: pip install langchain-ollama")
    Ollama = None
    ChatOllama = None

@dataclass
class NewsArticle:
    """Data structure for news articles"""
    title: str
    summary: str
    link: str
    published: str
    source: str
    category: str = "general"
    content: str = ""

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "model_provider": "openai",  # openai, ollama, lmstudio
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
                "provider": "huggingface",  # openai, huggingface
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
                "summary_length": "medium",  # short, medium, long
                "include_sources": True,
                "output_format": "both"  # markdown, json, both
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logging.warning(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

class NewsCollector:
    """Collects news from RSS feeds"""
    
    def __init__(self, config: ConfigManager):
        self.config = config.config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'News-RAG-Summarizer/1.0'
        })
    
    def fetch_rss_feed(self, url: str, source_name: str) -> List[NewsArticle]:
        """Fetch and parse RSS feed"""
        articles = []
        
        try:
            logging.info(f"Fetching RSS feed: {source_name}")
            
            # Use requests session for better control
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                logging.warning(f"Feed parsing warning for {source_name}: {feed.bozo_exception}")
            
            max_articles = self.config['output']['max_articles_per_source']
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Extract article data
                    title = entry.get('title', 'No Title')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')
                    
                    # Parse published date
                    published = ''
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6]).isoformat()
                    elif hasattr(entry, 'published'):
                        published = entry.published
                    
                    # Clean HTML from summary
                    if summary:
                        import re
                        summary = re.sub(r'<[^>]+>', '', summary)
                        summary = summary.strip()
                    
                    article = NewsArticle(
                        title=title,
                        summary=summary,
                        link=link,
                        published=published,
                        source=source_name,
                        content=summary  # For now, use summary as content
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logging.error(f"Error processing article from {source_name}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error fetching RSS feed {source_name}: {e}")
        
        logging.info(f"Collected {len(articles)} articles from {source_name}")
        return articles
    
    def collect_all_news(self) -> List[NewsArticle]:
        """Collect news from all configured RSS feeds"""
        all_articles = []
        
        for source_name, feed_url in self.config['rss_feeds'].items():
            articles = self.fetch_rss_feed(feed_url, source_name)
            all_articles.extend(articles)
            
            # Small delay between requests
            time.sleep(0.5)
        
        logging.info(f"Total articles collected: {len(all_articles)}")
        return all_articles

class LLMManager:
    """Manages different LLM providers"""
    
    def __init__(self, config: ConfigManager):
        self.config = config.config
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration"""
        provider = self.config['model_provider'].lower()
        
        if provider == 'openai':
            api_key = self.config['openai']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided in config or environment")
            
            return ChatOpenAI(
                openai_api_key=api_key,
                model=self.config['openai']['model'],
                temperature=self.config['openai']['temperature']
            )
        
        elif provider == 'ollama':
            if not Ollama:
                raise ImportError("Ollama not available. Install with: pip install langchain-ollama")
            
            return Ollama(
                base_url=self.config['ollama']['base_url'],
                model=self.config['ollama']['model'],
                temperature=self.config['ollama']['temperature']
            )
        
        elif provider == 'lmstudio':
            # LMStudio uses OpenAI-compatible API
            return ChatOpenAI(
                openai_api_base=self.config['lmstudio']['base_url'],
                openai_api_key=self.config['lmstudio']['api_key'],
                model=self.config['lmstudio']['model'],
                temperature=self.config['lmstudio']['temperature']
            )
        
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        provider = self.config['embeddings']['provider'].lower()
        
        if provider == 'openai':
            api_key = self.config['openai']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            return OpenAIEmbeddings(openai_api_key=api_key)
        
        else:  # Default to HuggingFace
            model_name = self.config['embeddings']['model']
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )

class RAGProcessor:
    """Processes news articles using RAG"""
    
    def __init__(self, llm_manager: LLMManager, config: ConfigManager):
        self.llm_manager = llm_manager
        self.config = config.config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def create_vector_store(self, articles: List[NewsArticle]) -> FAISS:
        """Create vector store from articles"""
        logging.info("Creating vector store from articles...")
        
        # Convert articles to documents
        documents = []
        for article in articles:
            content = f"Title: {article.title}\n\n"
            content += f"Source: {article.source}\n"
            content += f"Published: {article.published}\n\n"
            content += f"Content: {article.summary or article.content}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'title': article.title,
                    'source': article.source,
                    'link': article.link,
                    'published': article.published,
                    'category': article.category
                }
            )
            documents.append(doc)
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            splits,
            self.llm_manager.embeddings
        )
        
        logging.info(f"Vector store created with {len(splits)} document chunks")
        return vectorstore
    
    def create_summary_chain(self, vectorstore: FAISS):
        """Create the RAG chain for summarization"""
        
        # Define prompt template for detailed analysis
        template = """
        You are a senior news analyst and journalist. Based on the following news articles, provide an extremely comprehensive and detailed summary.
        
        Context from recent news:
        {context}
        
        Question: {question}
        
        Please provide a VERY DETAILED and EXTENSIVE summary that:
        1. Covers ALL major stories and developments in depth
        2. Provides detailed background context for each story
        3. Explains the significance and implications of each development
        4. Includes specific details, quotes, and data points from the articles
        5. Groups related stories together with thorough analysis
        6. Discusses potential future implications and consequences
        7. Mentions all credible sources and provides attribution
        8. Uses a journalistic writing style with rich detail
        9. Aims for comprehensive coverage - do not be brief or concise
        10. Include multiple paragraphs for each major topic area
        
        Write as if this is a comprehensive news briefing document that will be the primary source of information for decision-makers. Be thorough, detailed, and analytical.
        
        Summary:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain with more documents for richer context
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_manager.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 25}  # Increased from 10 to 25 for more context
            ),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def generate_summary(self, articles: List[NewsArticle]) -> str:
        """Generate comprehensive news summary using RAG"""
        if not articles:
            return "No news articles found to summarize."
        
        logging.info("Generating comprehensive news summary using RAG...")
        
        # Create vector store
        vectorstore = self.create_vector_store(articles)
        
        # Create summary chain
        qa_chain = self.create_summary_chain(vectorstore)
        
        # Generate multiple detailed sections
        areas_of_interest = self.config['areas_of_interest']
        
        # Create a comprehensive query for detailed analysis
        main_query = f"""
        Please provide an extremely comprehensive and detailed news analysis covering ALL of these areas: {', '.join(areas_of_interest)}.
        
        For each major story or topic area, provide:
        - Detailed background and context
        - Current developments and latest updates
        - Key players and stakeholders involved
        - Specific facts, figures, and quotes from the articles
        - Analysis of implications and significance
        - Potential future developments or consequences
        - Multiple perspectives where relevant
        
        Cover ALL significant stories from the provided articles. This should be a thorough, in-depth analysis suitable for a comprehensive news briefing. Do not summarize briefly - provide extensive detail and analysis for each topic.
        
        Structure your response with clear sections for different topic areas, and within each section, provide multiple paragraphs of detailed coverage.
        """
        
        try:
            logging.info("Generating main comprehensive summary...")
            main_summary = qa_chain.invoke({
                "query": main_query
            })
            
            # Generate additional focused summaries for major topic areas
            focused_summaries = []
            major_topics = ["world news", "politics", "technology", "business", "health", "climate"]
            
            for topic in major_topics:
                if topic in [area.lower() for area in areas_of_interest]:
                    logging.info(f"Generating focused summary for: {topic}")
                    topic_query = f"""
                    Provide an extremely detailed analysis specifically focused on {topic.upper()} stories from the news articles.
                    
                    Include:
                    - All relevant stories and developments in {topic}
                    - Detailed background context for each story
                    - Specific facts, data, and quotes
                    - Analysis of trends and patterns
                    - Implications and significance
                    - Future outlook and potential developments
                    
                    Be comprehensive and detailed - this is a specialized {topic} briefing section.
                    """
                    
                    try:
                        topic_summary = qa_chain.invoke({
                            "query": topic_query
                        })
                        focused_summaries.append(f"\n\n## DETAILED {topic.upper()} ANALYSIS\n\n{topic_summary['result']}")
                    except Exception as e:
                        logging.warning(f"Error generating {topic} summary: {e}")
                        continue
            
            # Combine all summaries
            full_summary = main_summary["result"]
            if focused_summaries:
                full_summary += "\n\n" + "".join(focused_summaries)
            
            return full_summary
        
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

class OutputManager:
    """Manages output generation and formatting"""
    
    def __init__(self, config: ConfigManager):
        self.config = config.config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_summary_markdown(self, summary: str, articles: List[NewsArticle]) -> str:
        """Save summary as Markdown file"""
        filename = f"news_summary_{self.timestamp}.md"
        
        content = f"""# News Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Areas of Interest
{', '.join(self.config['areas_of_interest'])}

## Summary
{summary}

## Sources
- Total articles processed: {len(articles)}
- Sources: {', '.join(set(article.source for article in articles))}

---
*Generated by News RAG Summarizer*
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"Markdown summary saved to: {filename}")
        return filename
    
    def save_summary_json(self, summary: str, articles: List[NewsArticle]) -> str:
        """Save summary and articles as JSON file"""
        filename = f"news_data_{self.timestamp}.json"
        
        data = {
            "metadata": {
                "generated_on": datetime.now().isoformat(),
                "areas_of_interest": self.config['areas_of_interest'],
                "total_articles": len(articles),
                "sources": list(set(article.source for article in articles))
            },
            "summary": summary,
            "articles": [asdict(article) for article in articles]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"JSON data saved to: {filename}")
        return filename
    
    def create_download_package(self, summary: str, articles: List[NewsArticle]) -> str:
        """Create a zip package with all outputs"""
        import zipfile
        
        zip_filename = f"news_summary_package_{self.timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Add markdown summary
            if self.config['output']['output_format'] in ['markdown', 'both']:
                md_file = self.save_summary_markdown(summary, articles)
                zipf.write(md_file)
                os.remove(md_file)  # Clean up temporary file
            
            # Add JSON data
            if self.config['output']['output_format'] in ['json', 'both']:
                json_file = self.save_summary_json(summary, articles)
                zipf.write(json_file)
                os.remove(json_file)  # Clean up temporary file
            
            # Add configuration file
            config_content = json.dumps(self.config, indent=2)
            zipf.writestr("config_used.json", config_content)
            
            # Add README
            readme_content = """# News RAG Summarizer Output

This package contains:
- news_summary_*.md: Markdown formatted summary
- news_data_*.json: Complete data including all articles
- config_used.json: Configuration used for this run

## Usage
The markdown file contains a human-readable summary.
The JSON file contains structured data for further processing.

Generated by News RAG Summarizer
"""
            zipf.writestr("README.txt", readme_content)
        
        logging.info(f"Download package created: {zip_filename}")
        return zip_filename

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('news_rag.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main application function"""
    print("🗞️  News RAG Summarizer")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        news_collector = NewsCollector(config_manager)
        llm_manager = LLMManager(config_manager)
        rag_processor = RAGProcessor(llm_manager, config_manager)
        output_manager = OutputManager(config_manager)
        
        print(f"📊 Model Provider: {config_manager.config['model_provider']}")
        print(f"🎯 Areas of Interest: {', '.join(config_manager.config['areas_of_interest'][:5])}...")
        print(f"📰 RSS Sources: {len(config_manager.config['rss_feeds'])}")
        print()
        
        # Collect news articles
        print("🔄 Collecting news articles...")
        articles = news_collector.collect_all_news()
        
        if not articles:
            print("❌ No articles collected. Check RSS feeds and network connection.")
            return
        
        print(f"✅ Collected {len(articles)} articles")
        
        # Generate summary using RAG
        print("🤖 Generating summary using RAG...")
        summary = rag_processor.generate_summary(articles)
        
        # Save outputs
        print("💾 Saving outputs...")
        package_file = output_manager.create_download_package(summary, articles)
        
        print(f"🎉 Complete! Download package: {package_file}")
        print(f"📝 Log file: news_rag.log")
        
        # Display brief summary
        print("\n📋 Brief Summary:")
        print("-" * 30)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

