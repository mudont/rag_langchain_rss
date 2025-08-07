# Implementation Plan

- [x] 1. Modify OutputManager class initialization
  - Update `__init__` method to include output directory path
  - Create `self.output_dir` attribute with format `news/{timestamp}`
  - _Requirements: 1.1_

- [x] 2. Create directory creation method
  - Implement `create_output_directory()` method to replace `create_download_package()`
  - Add logic to create `news/` parent directory if it doesn't exist
  - Add logic to create timestamped subdirectory
  - Include proper error handling for directory creation failures
  - _Requirements: 1.1, 3.1_

- [x] 3. Modify markdown file saving method
  - Update `save_summary_markdown()` to save files in the output directory
  - Change file path to use `self.output_dir` instead of current directory
  - Ensure the method returns the full file path for logging
  - _Requirements: 2.1, 3.2_

- [x] 4. Modify JSON file saving method
  - Update `save_summary_json()` to save files in the output directory
  - Change file path to use `self.output_dir` instead of current directory
  - Ensure the method returns the full file path for logging
  - _Requirements: 2.2, 3.2_

- [x] 5. Implement configuration and README file creation
  - Add method to save `config_used.json` in the output directory
  - Add method to create `README.txt` explaining directory contents
  - Ensure both files are saved to the timestamped directory
  - _Requirements: 2.3, 2.4_

- [x] 6. Update main output creation workflow
  - Replace `create_download_package()` calls with `create_output_directory()`
  - Update the method to call all file saving methods and save to directory
  - Remove zip file creation logic entirely
  - Return directory path instead of zip filename
  - _Requirements: 1.3, 2.1, 2.2, 2.3, 2.4_

- [ ] 7. Update console output and logging
  - Modify main function to display directory path instead of zip filename
  - Update logging messages to reference directories instead of zip files
  - Add logging for directory creation success
  - Update final success message to show directory location
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 8. Add error handling and validation
  - Add try-catch blocks around directory creation
  - Implement fallback behavior if directory creation fails
  - Add validation that all expected files were created
  - Log any errors during file creation process
  - _Requirements: 1.1, 3.1_

- [ ] 9. Test the implementation
  - Run the news summarizer to verify directory creation works
  - Check that all files are created in the correct timestamped directory
  - Verify console output shows correct directory path
  - Test multiple runs to ensure unique directories are created
  - _Requirements: 1.4, 3.3_

- [ ] 10. Clean up and finalize
  - Remove any unused zip-related imports
  - Remove the old `create_download_package()` method
  - Update any remaining references to zip files in comments or docstrings
  - Verify no zip files are created during execution
  - _Requirements: 1.3_