# PDF Parser and Structured JSON Extractor

A comprehensive Python program that parses PDF files and extracts their content into well-structured JSON format. The program preserves hierarchical organization of documents and clearly identifies different data types including paragraphs, tables, and charts.

## Features

- **Multi-library approach**: Combines PyMuPDF, pdfplumber, and Camelot for robust content extraction
- **Hierarchical structure preservation**: Automatically detects sections and sub-sections based on font sizes and formatting
- **Content type identification**: Distinguishes between paragraphs, tables, and charts/images
- **Table extraction**: Uses multiple extraction methods (Camelot lattice/stream, pdfplumber) for maximum accuracy
- **Chart detection**: Basic image analysis to identify potential charts and diagrams
- **Clean JSON output**: Well-structured JSON with page-level hierarchy and content metadata
- **Command-line interface**: Easy-to-use CLI with debugging options
- **Robust error handling**: Graceful handling of various PDF formats and extraction failures

## Requirements

### System Dependencies

Before installing Python packages, ensure you have the following system dependencies:

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y python3-tk ghostscript libgl1-mesa-glx libglib2.0-0
```

#### macOS:
```bash
brew install ghostscript
```

#### Windows:
- Download and install Ghostscript from: https://www.ghostscript.com/download/gsdnld.html
- Add Ghostscript to your system PATH

### Python Requirements

- Python 3.8 or higher
- See `requirements.txt` for detailed package dependencies

## Installation

1. **Clone or download the project files:**
   ```bash
   # If using git
   git clone <repository-url>
   cd pdf-parser

   # Or simply download the files to a directory
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv pdf_parser_env

   # Activate the virtual environment
   # On Windows:
   pdf_parser_env\Scripts\activate
   # On macOS/Linux:
   source pdf_parser_env/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python pdf_parser.py --help
   ```

## Getting Started (Quick Start)

Follow these copy-paste commands to get the project running quickly.

Windows PowerShell
```powershell
cd C:\Users\Eyosi\projects\jupyter-work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux
```bash
cd /path/to/your/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- On Windows ensure `python` is the same interpreter you used to create the virtualenv.
- Install system dependencies (Ghostscript, OpenCV libs) before installing Camelot-related extras.


## Usage

### Basic Usage

```bash
python pdf_parser.py input.pdf output.json
```

### Command Line Options

- `input_pdf`: Path to the input PDF file (required)
- `output_json`: Path to the output JSON file (required)
- `--debug`: Enable debug logging for troubleshooting
- `--compact`: Generate compact JSON without indentation (saves space)

### Examples

1. **Basic extraction:**
   ```bash
   python pdf_parser.py document.pdf extracted_content.json
   ```

2. **With debug output:**
   ```bash
   python pdf_parser.py document.pdf extracted_content.json --debug
   ```

3. **Compact JSON output:**
   ```bash
   python pdf_parser.py document.pdf extracted_content.json --compact
   ```

### Programmatic Usage

```python
from pdf_parser import PDFContentExtractor

# Using context manager (recommended)
with PDFContentExtractor("document.pdf", debug=True) as extractor:
    content = extractor.extract_to_json()
    extractor.save_json("output.json")

# Or manual resource management
extractor = PDFContentExtractor("document.pdf")
try:
    content = extractor.extract_to_json()
    # Process content...
finally:
    extractor.close()
```

## Run The Parser (CLI)

Basic usage (PowerShell / Linux / macOS):

```powershell
python pdf_parser.py "C:\path\to\document.pdf" output.json
```

With debug logging (prints detailed extraction steps):

```powershell
python pdf_parser.py document.pdf output.json --debug
```

Generate compact JSON (no indentation):

```powershell
python pdf_parser.py document.pdf output.json --compact
```

If the `input_pdf` path does not exist the script will open a file-picker dialog (GUI) so you can select a PDF interactively.

## Running Tests and CI

Run the included sample JSON test locally:

```bash
pip install pytest
pytest -q
```

The test suite includes a small, self-contained test `test_create_sample_json` which creates `sample_output.json` and validates its structure — this does not require a real PDF and is safe to run in CI.

Suggested GitHub Actions workflow (minimal):

1. Create `.github/workflows/ci.yml` with steps to:
   - Check out code
   - Set up Python 3.9+
   - Install `pip install -r requirements.txt`
   - Run `pytest -q`

This will ensure syntax and the sample test run on each push.


## JSON Output Structure

The program generates JSON output with the following structure:

```json
{
  "document_info": {
    "filename": "example.pdf",
    "total_pages": 10,
    "extraction_timestamp": "2025-09-18T12:47:00"
  },
  "pages": [
    {
      "page_number": 1,
      "content": [
        {
          "type": "paragraph",
          "section": "Introduction",
          "sub_section": "Background",
          "text": "This is an example paragraph extracted from the PDF..."
        },
        {
          "type": "table",
          "section": "Financial Data",
          "description": "Table extracted from page 1",
          "table_data": [
            ["Year", "Revenue", "Profit"],
            ["2022", "$10M", "$2M"],
            ["2023", "$12M", "$3M"]
          ]
        },
        {
          "type": "chart",
          "section": "Performance Overview",
          "description": "Chart detected on page 1",
          "table_data": [
            ["X_Label", "Y_Label"],
            ["2022", "$10M"],
            ["2023", "$12M"]
          ]
        }
      ]
    }
  ]
}
```

### Field Descriptions

- **document_info**: Metadata about the processed document
- **pages**: Array of page objects, each containing:
  - **page_number**: 1-indexed page number
  - **content**: Array of content items, each with:
    - **type**: Content type (`"paragraph"`, `"table"`, or `"chart"`)
    - **section**: Main section heading (if detected)
    - **sub_section**: Sub-section heading (if applicable)
    - **text**: Text content (for paragraphs)
    - **description**: Content description (for tables/charts)
    - **table_data**: 2D array of table/chart data

## Architecture and Implementation

### Core Components

1. **PDFContentExtractor**: Main class that orchestrates the extraction process
2. **Section Detection**: Font-size based hierarchy detection for headings
3. **Table Extraction**: Multi-method approach using Camelot and pdfplumber
4. **Chart Detection**: Basic image analysis for chart identification
5. **Content Organization**: Hierarchical content structuring and JSON formatting

### Library Usage Strategy

- **PyMuPDF (fitz)**: Primary library for text extraction, font analysis, and image detection
- **pdfplumber**: Backup table extraction and detailed text positioning
- **Camelot**: Specialized table extraction with lattice and stream methods
- **OpenCV + PIL**: Image processing for chart detection
- **Pandas/NumPy**: Data manipulation and analysis

### Section Detection Algorithm

The program detects document hierarchy using:

1. **Font size analysis**: Identifies heading levels based on font sizes relative to body text
2. **Text pattern matching**: Recognizes common heading patterns (numbered sections, title case, etc.)
3. **Position-based inference**: Uses text positioning to determine section context
4. **Heuristic filtering**: Removes false positives using length and punctuation rules

## Troubleshooting

### Common Issues

1. **Ghostscript not found error:**
   - Ensure Ghostscript is installed and added to your system PATH
   - On Windows, restart your command prompt after installation

2. **Camelot extraction fails:**
   - The program automatically falls back to pdfplumber for table extraction
   - Enable debug mode to see detailed error messages

3. **Memory issues with large PDFs:**
   - Process PDFs in smaller chunks if memory is limited
   - Consider using the `--compact` flag to reduce memory usage

4. **Poor table extraction quality:**
   - Try different PDF files to compare results
   - Some scanned PDFs may require OCR preprocessing

5. **Charts not detected:**
   - Current chart detection is basic and may miss complex visualizations
   - Future versions will include more sophisticated ML-based detection

### Debug Mode

Enable debug logging to troubleshoot issues:

```bash
python pdf_parser.py document.pdf output.json --debug
```

This will provide detailed information about:
- PDF parsing progress
- Content extraction steps
- Error messages with stack traces
- Performance timing information

## Limitations and Future Enhancements

### Current Limitations

- **Chart data extraction**: Currently provides placeholder data for detected charts
- **OCR support**: No built-in OCR for scanned documents
- **Complex layouts**: May struggle with multi-column or complex page layouts
- **Language support**: Optimized for English text patterns

### Planned Enhancements

- **Advanced chart analysis**: ML-based chart type detection and data extraction
- **OCR integration**: Support for scanned documents using Tesseract
- **Layout analysis**: Better handling of complex document layouts
- **Performance optimization**: Parallel processing for large documents
- **Format support**: Extended support for additional document formats

## Contributing

Contributions are welcome! Please consider the following areas:

- Enhanced chart detection algorithms
- Better section hierarchy detection
- Performance optimizations
- Additional output formats
- Test coverage improvements

## License

This project is provided as-is for educational and commercial use. Please ensure compliance with all third-party library licenses.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Enable debug mode for detailed error information
3. Verify all system dependencies are correctly installed
4. Test with different PDF files to isolate document-specific issues

---

**Note**: This tool is designed for structured documents with clear hierarchies. Results may vary depending on the PDF format, creation method, and document complexity.
