# Requirements Document

## Introduction

This feature modifies the news RAG summarizer to organize output files in timestamped subdirectories instead of creating zip packages. This will provide better organization and easier access to generated news summaries and data files.

## Requirements

### Requirement 1

**User Story:** As a user running the news summarizer, I want output files to be organized in timestamped directories, so that I can easily browse and access different runs without dealing with zip files.

#### Acceptance Criteria

1. WHEN the news summarizer runs THEN the system SHALL create a subdirectory under `news/` with format `YYYYMMDD_HHMMSS`
2. WHEN output files are generated THEN the system SHALL save all files directly in the timestamped subdirectory
3. WHEN the run completes THEN the system SHALL NOT create any zip files
4. WHEN multiple runs occur THEN each run SHALL create its own unique timestamped directory

### Requirement 2

**User Story:** As a user, I want the same output files as before, so that I don't lose any functionality when switching from zip to directory output.

#### Acceptance Criteria

1. WHEN files are saved to the directory THEN the system SHALL include the markdown summary file
2. WHEN files are saved to the directory THEN the system SHALL include the JSON data file  
3. WHEN files are saved to the directory THEN the system SHALL include the configuration used for the run
4. WHEN files are saved to the directory THEN the system SHALL include a README file explaining the contents

### Requirement 3

**User Story:** As a user, I want clear feedback about where files are saved, so that I know where to find the generated content.

#### Acceptance Criteria

1. WHEN the directory is created THEN the system SHALL log the directory path
2. WHEN files are saved THEN the system SHALL display the directory location in the console output
3. WHEN the run completes THEN the system SHALL show a summary of files created in the directory