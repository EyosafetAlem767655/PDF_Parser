#!/usr/bin/env python3
"""
Test script for PDF Parser
==========================

This script demonstrates how to use the PDF Parser programmatically
and provides basic testing functionality.
"""

import json
import sys
from pathlib import Path
from pdf_parser import PDFContentExtractor


def test_extraction(pdf_path: str, output_path: str = None):
    """
    Test PDF extraction with a given file.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path for JSON
    """
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        return False

    try:
        print(f"Testing extraction with: {pdf_path}")

        with PDFContentExtractor(pdf_path, debug=True) as extractor:
            # Extract content
            content = extractor.extract_to_json()

            # Print summary
            print(f"\nExtraction Summary:")
            print(f"- Document: {content['document_info']['filename']}")
            print(f"- Pages: {content['document_info']['total_pages']}")
            print(f"- Total content items: {sum(len(page['content']) for page in content['pages'])}")

            # Count content types
            content_types = {}
            for page in content['pages']:
                for item in page['content']:
                    content_type = item.get('type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            print(f"- Content breakdown: {dict(content_types)}")

            # Save if output path provided
            if output_path:
                extractor.save_json(output_path)
                print(f"\nOutput saved to: {output_path}")

            return True

    except Exception as e:
        print(f"Error during extraction: {e}")
        return False


def create_sample_json():
    """Create a sample JSON output for reference."""
    sample_data = {
        "document_info": {
            "filename": "sample.pdf",
            "total_pages": 2,
            "extraction_timestamp": "2025-09-18T12:47:00"
        },
        "pages": [
            {
                "page_number": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "section": "Introduction",
                        "sub_section": "Overview",
                        "text": "This is a sample paragraph showing how the PDF parser structures extracted text content with proper section hierarchy."
                    },
                    {
                        "type": "table",
                        "section": "Data Analysis",
                        "description": "Sample table extracted from PDF",
                        "table_data": [
                            ["Quarter", "Revenue", "Growth"],
                            ["Q1 2023", "$100M", "5%"],
                            ["Q2 2023", "$110M", "10%"],
                            ["Q3 2023", "$120M", "9%"]
                        ]
                    }
                ]
            },
            {
                "page_number": 2,
                "content": [
                    {
                        "type": "paragraph",
                        "section": "Conclusion",
                        "sub_section": None,
                        "text": "This concluding paragraph demonstrates how the parser maintains section context across pages."
                    },
                    {
                        "type": "chart",
                        "section": "Appendix",
                        "description": "Chart showing quarterly performance trends",
                        "table_data": [
                            ["Quarter", "Revenue"],
                            ["Q1", "100"],
                            ["Q2", "110"],
                            ["Q3", "120"]
                        ]
                    }
                ]
            }
        ]
    }

    with open('sample_output.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print("Sample JSON output created: sample_output.json")


def main():
    """Main test function."""
    print("PDF Parser Test Script")
    print("=====================")

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_parser.py <pdf_file> [output_file]")
        print("  python test_parser.py --sample")
        print("\nExamples:")
        print("  python test_parser.py document.pdf")
        print("  python test_parser.py document.pdf output.json")
        print("  python test_parser.py --sample  # Creates sample JSON")
        return

    if sys.argv[1] == '--sample':
        create_sample_json()
        return

    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = test_extraction(pdf_file, output_file)
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
