# Design Document

## Overview

This design modifies the `OutputManager` class in the news RAG summarizer to save output files in timestamped subdirectories under `news/` instead of creating zip packages. The change provides better file organization and easier access to generated content.

## Architecture

The modification focuses on the `OutputManager` class, specifically replacing the `create_download_package()` method with a new `create_output_directory()` method that:

1. Creates a timestamped directory structure
2. Saves files directly to the directory
3. Provides clear feedback about file locations

## Components and Interfaces

### Modified OutputManager Class

```python
class OutputManager:
    def __init__(self, config: ConfigManager):
        self.config = config.config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"news/{self.timestamp}"
    
    def create_output_directory(self, summary: str, articles: List[NewsArticle]) -> str:
        """Create timestamped directory with all outputs"""
        # Creates news/YYYYMMDD_HHMMSS/ directory
        # Saves all files directly to this directory
        # Returns the directory path
    
    def save_summary_markdown(self, summary: str, articles: List[NewsArticle]) -> str:
        """Save summary as Markdown file in output directory"""
        # Modified to save to self.output_dir
    
    def save_summary_json(self, summary: str, articles: List[NewsArticle]) -> str:
        """Save summary and articles as JSON file in output directory"""
        # Modified to save to self.output_dir
```

### Directory Structure

```
news/
├── 20250806_212345/
│   ├── news_summary_20250806_212345.md
│   ├── news_data_20250806_212345.json
│   ├── config_used.json
│   └── README.txt
├── 20250806_213456/
│   ├── news_summary_20250806_213456.md
│   ├── news_data_20250806_213456.json
│   ├── config_used.json
│   └── README.txt
└── ...
```

## Data Models

### File Naming Convention
- Directory: `news/YYYYMMDD_HHMMSS/`
- Markdown: `news_summary_YYYYMMDD_HHMMSS.md`
- JSON: `news_data_YYYYMMDD_HHMMSS.json`
- Config: `config_used.json`
- README: `README.txt`

### Output Directory Contents
1. **Markdown Summary**: Human-readable news summary
2. **JSON Data**: Complete structured data with all articles
3. **Configuration**: Settings used for the run
4. **README**: Explanation of directory contents

## Error Handling

### Directory Creation
- Check if `news/` directory exists, create if needed
- Handle permissions errors when creating directories
- Provide fallback to current directory if `news/` creation fails

### File Writing
- Handle file writing permissions errors
- Ensure atomic writes where possible
- Log all file operations for debugging

### Timestamp Conflicts
- Use microseconds if same-second runs occur
- Implement retry logic for directory creation conflicts

## Testing Strategy

### Unit Tests
1. **Directory Creation**: Test that timestamped directories are created correctly
2. **File Saving**: Verify all files are saved to the correct directory
3. **Error Handling**: Test behavior when directory creation fails
4. **Timestamp Uniqueness**: Ensure unique directories for concurrent runs

### Integration Tests
1. **End-to-End**: Run complete news summarizer and verify directory structure
2. **Multiple Runs**: Test that multiple runs create separate directories
3. **File Content**: Verify file contents match expected format

### Manual Testing
1. **Directory Navigation**: Verify easy browsing of output directories
2. **File Access**: Confirm files can be opened directly without extraction
3. **Console Output**: Check that directory paths are clearly displayed

## Implementation Notes

### Backward Compatibility
- Remove zip file creation entirely
- Update console messages to reference directories instead of zip files
- Maintain same file formats and content structure

### Performance Considerations
- Directory creation is faster than zip compression
- Files are immediately accessible without extraction
- Reduced disk I/O compared to zip creation and cleanup

### Configuration Changes
- No configuration changes required
- Existing `output_format` setting still applies to file types created
- Directory structure is consistent regardless of configuration