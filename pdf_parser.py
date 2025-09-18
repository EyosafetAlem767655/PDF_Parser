#!/usr/bin/env python3
"""
PDF Parser and Structured JSON Extractor
=========================================

A comprehensive Python program that parses PDF files and extracts content into 
well-structured JSON format, preserving hierarchical organization and identifying 
different content types (paragraphs, tables, charts).

"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core PDF processing libraries
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# For chart/image detection
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter, defaultdict
import re


class PDFContentExtractor:
    """
    Main class for extracting structured content from PDF files.

    This class combines multiple PDF processing libraries to achieve comprehensive
    content extraction including text, tables, and chart detection.
    """

    def __init__(self, pdf_path: str, debug: bool = False):
        """
        Initialize the PDF extractor.

        Args:
            pdf_path (str): Path to the PDF file
            debug (bool): Enable debug logging
        """
        self.pdf_path = Path(pdf_path)
        self.debug = debug
        self._setup_logging()

        # Initialize PDF documents with different libraries
        self.pymupdf_doc = None
        self.pdfplumber_doc = None

        self._open_documents()

    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _open_documents(self):
        """Open PDF documents with different libraries."""
        try:
            self.pymupdf_doc = fitz.open(self.pdf_path)
            self.pdfplumber_doc = pdfplumber.open(self.pdf_path)
            self.logger.info(f"Successfully opened PDF: {self.pdf_path}")
        except Exception as e:
            self.logger.error(f"Error opening PDF: {e}")
            raise

    def _get_page_text_lines(self, fitz_page) -> List[str]:
        """Return a list of text lines for a page (preserves order)."""
        text = fitz_page.get_text("text") or ""
        # Split on lines and normalize whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines

    def _detect_repeated_lines(self, pages_lines: List[List[str]], min_pages_fraction: float = 0.6) -> set:
        """
        Detect lines that repeat across many pages (likely headers/footers).

        Args:
            pages_lines: list where each entry is the list of lines for a page
            min_pages_fraction: fraction of pages a line must appear in to be considered repeated

        Returns:
            set of repeated line strings
        """
        counter = Counter()
        total = len(pages_lines) if pages_lines else 1
        # Consider only top/bottom best candidate lines per page
        for lines in pages_lines:
            candidates = []
            if lines:
                candidates.extend(lines[:3])
                candidates.extend(lines[-3:])
            unique_cands = set(candidates)
            for c in unique_cands:
                counter[c] += 1

        threshold = max(1, int(total * min_pages_fraction))
        repeated = {line for line, cnt in counter.items() if cnt >= threshold and line}
        if self.debug:
            self.logger.debug(f"Detected repeated header/footer lines (threshold={threshold}): {repeated}")
        return repeated

    def _normalize_text(self, text: str) -> str:
        """Normalize and reflow extracted text: collapse internal linebreaks and extra spaces."""
        if not text:
            return text
        # Replace sequences of whitespace/newlines with a single space
        t = re.sub(r'\s*\n\s*', ' ', text)
        t = re.sub(r'[ \t]{2,}', ' ', t)
        t = t.strip()
        return t

    def _clean_table(self, table: List[List[Any]]) -> List[List[Any]]:
        """Clean up common table extraction artifacts.

        - Remove an initial numeric index row like [0,1,2,...]
        - Remove rows that are entirely empty
        - Remove columns that are entirely empty
        """
        if not table:
            return table

        # Defensive copy
        tbl = [list(row) for row in table]

        # If first row is sequential integers (0..n-1), drop it
        try:
            first = tbl[0]
            if all(isinstance(x, (int, np.integer)) for x in first) or all(re.match(r'^\d+$', str(x)) for x in first):
                tbl = tbl[1:]
        except Exception:
            pass

        # Remove empty-only rows
        tbl = [row for row in tbl if any(str(cell).strip() for cell in row)]
        if not tbl:
            return tbl

        # Remove empty-only columns
        cols = list(zip(*tbl))
        keep_idx = [i for i, col in enumerate(cols) if any(str(c).strip() for c in col)]
        if not keep_idx:
            return []
        cleaned = [[row[i] for i in keep_idx] for row in tbl]
        return cleaned

    def _detect_section_hierarchy(self, page_text_blocks: List[Dict]) -> Dict[str, Any]:
        """
        Detect section hierarchy based on font sizes and formatting.

        Args:
            page_text_blocks: List of text blocks with formatting information

        Returns:
            Dictionary mapping text to section levels
        """
        font_sizes = []
        text_blocks = []

        for block in page_text_blocks:
            if 'size' in block and 'text' in block:
                font_sizes.append(block['size'])
                text_blocks.append(block)

        if not font_sizes:
            return {}

        # Determine body text size (most common)
        font_counter = Counter(font_sizes)
        body_size = font_counter.most_common(1)[0][0]

        # Create hierarchy mapping
        unique_sizes = sorted(set(font_sizes), reverse=True)
        hierarchy = {}

        for block in text_blocks:
            text = block['text'].strip()
            size = block['size']

            # Skip if text is too short or appears to be body text
            if len(text) < 3 or size <= body_size:
                continue

            # Determine section level based on font size
            if size > body_size + 4:
                level = 1  # H1
            elif size > body_size + 2:
                level = 2  # H2
            else:
                level = 3  # H3

            # Additional heuristics for section detection
            if (self._is_likely_heading(text) and 
                text not in hierarchy):
                hierarchy[text] = {
                    'level': level,
                    'font_size': size
                }

        return hierarchy

    def _is_likely_heading(self, text: str) -> bool:
        """
        Determine if text is likely a heading based on various heuristics.

        Args:
            text: Text to analyze

        Returns:
            Boolean indicating if text is likely a heading
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Too short or too long
        if len(text) < 3 or len(text) > 200:
            return False

        # Ends with punctuation (likely not a heading)
        if text.endswith(('.', ':', ';', '!', '?')):
            return False

        # Contains too many numbers (likely not a heading)
        if sum(c.isdigit() for c in text) / len(text) > 0.5:
            return False

        # Common heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',    # All caps
            r'^[A-Z][a-z]+.*[A-Z]', # Title case
            r'^(Chapter|Section|Part)\s+\d+',  # Explicit chapters/sections
        ]

        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _extract_tables_camelot(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot library.

        Args:
            page_num: Page number (1-indexed)

        Returns:
            List of table dictionaries
        """
        tables = []
        try:
            # Extract tables using both lattice and stream methods
            camelot_tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages=str(page_num),
                flavor='lattice'
            )

            if camelot_tables.n == 0:
                # Try stream method if lattice fails
                camelot_tables = camelot.read_pdf(
                    str(self.pdf_path), 
                    pages=str(page_num),
                    flavor='stream'
                )

            for i, table in enumerate(camelot_tables):
                if table.df is not None and not table.df.empty:
                    # Convert DataFrame to list of lists
                    table_data = table.df.values.tolist()
                    # Add headers
                    headers = table.df.columns.tolist()
                    if headers and any(str(h).strip() for h in headers):
                        table_data.insert(0, headers)

                    tables.append({
                        'table_id': f'table_{page_num}_{i}',
                        'table_data': table_data,
                        'accuracy': table.accuracy if hasattr(table, 'accuracy') else None
                    })

        except Exception as e:
            self.logger.warning(f"Camelot table extraction failed for page {page_num}: {e}")

        return tables

    def _extract_tables_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber.

        Args:
            page: pdfplumber page object

        Returns:
            List of table dictionaries
        """
        tables = []
        try:
            plumber_tables = page.extract_tables()

            for i, table in enumerate(plumber_tables):
                if table and len(table) > 1:  # Ensure table has content
                    tables.append({
                        'table_id': f'table_{page.page_number}_{i}',
                        'table_data': table
                    })

        except Exception as e:
            self.logger.warning(f"Pdfplumber table extraction failed: {e}")

        return tables

    def _detect_charts_images(self, page_fitz, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect charts and images in the PDF page.

        Args:
            page_fitz: PyMuPDF page object
            page_num: Page number

        Returns:
            List of chart/image dictionaries
        """
        charts = []
        try:
            # Get image list from page
            image_list = page_fitz.get_images()

            for i, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(self.pymupdf_doc, xref)

                    if pix.n - pix.alpha < 4:  # Ensure it's not CMYK
                        # Convert to RGB if necessary
                        if pix.n - pix.alpha != 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Convert to PIL Image for analysis
                        img_data = pix.tobytes("ppm")
                        pil_img = Image.open(io.BytesIO(img_data))

                        # Basic chart detection heuristics
                        width, height = pil_img.size
                        aspect_ratio = width / height

                        # Analyze if image might be a chart
                        is_chart = self._analyze_chart_likelihood(pil_img)

                        chart_info = {
                            'type': 'chart' if is_chart else 'image',
                            'chart_id': f'chart_{page_num}_{i}',
                            'description': f"{'Chart' if is_chart else 'Image'} detected on page {page_num}",
                            'dimensions': {'width': width, 'height': height},
                            'aspect_ratio': round(aspect_ratio, 2)
                        }

                        # If it's likely a chart, try to extract some data
                        if is_chart:
                            chart_info['table_data'] = self._extract_chart_data_placeholder(pil_img)

                        charts.append(chart_info)

                    pix = None  # Cleanup

                except Exception as e:
                    self.logger.warning(f"Error processing image {i} on page {page_num}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Chart detection failed for page {page_num}: {e}")

        return charts

    def _analyze_chart_likelihood(self, pil_img: Image.Image) -> bool:
        """
        Analyze if an image is likely a chart based on visual characteristics.

        Args:
            pil_img: PIL Image object

        Returns:
            Boolean indicating if image is likely a chart
        """
        try:
            # Convert to numpy array for analysis
            img_array = np.array(pil_img)

            # Basic heuristics for chart detection
            height, width = img_array.shape[:2]

            # Charts typically have certain aspect ratios
            aspect_ratio = width / height
            if not (0.5 <= aspect_ratio <= 3.0):
                return False

            # Charts often have white backgrounds with colored elements
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Check for line-like structures (common in charts)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)

            # Charts typically have moderate edge density
            if 0.01 <= edge_density <= 0.3:
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Chart analysis failed: {e}")
            return False

    def _extract_chart_data_placeholder(self, pil_img: Image.Image) -> List[List[str]]:
        """
        Placeholder for chart data extraction.
        In a real implementation, this would use OCR or ML models.

        Args:
            pil_img: PIL Image object

        Returns:
            Placeholder table data
        """
        # Placeholder implementation
        return [
            ["X_Label", "Y_Label"],
            ["Category_1", "Value_1"],
            ["Category_2", "Value_2"]
        ]

    def _extract_page_content(self, page_num: int) -> Dict[str, Any]:
        """
        Extract all content from a single page.

        Args:
            page_num: Page number (0-indexed for internal use)

        Returns:
            Dictionary containing page content
        """
        page_content = {
            'page_number': page_num + 1,
            'content': []
        }

        try:
            # Get page objects
            fitz_page = self.pymupdf_doc[page_num]
            plumber_page = self.pdfplumber_doc.pages[page_num]

            # Build lines for all pages once (used to detect headers/footers)
            # We'll lazily compute repeated lines across the document on first call
            if not hasattr(self, '_cached_pages_lines'):
                self._cached_pages_lines = []
                for p in range(len(self.pymupdf_doc)):
                    self._cached_pages_lines.append(self._get_page_text_lines(self.pymupdf_doc[p]))
                self._repeated_lines = self._detect_repeated_lines(self._cached_pages_lines)

            # Extract text blocks with formatting information for hierarchy
            text_blocks = fitz_page.get_text("dict")

            all_text_blocks = []
            for block in text_blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_text_blocks.append({
                                'text': span.get('text', ''),
                                'size': span.get('size', 12),
                                'font': span.get('font', ''),
                                'flags': span.get('flags', 0)
                            })

            hierarchy = self._detect_section_hierarchy(all_text_blocks)

            # Extract paragraphs and normalize them
            paragraphs = self._extract_paragraphs(fitz_page, hierarchy)
            # Normalize paragraph text and filter header/footer lines
            cleaned_paragraphs = []
            for p in paragraphs:
                txt = self._normalize_text(p['text'])
                # filter out repeated header/footer lines
                if any(line in txt for line in self._repeated_lines):
                    continue
                p['text'] = txt
                cleaned_paragraphs.append(p)

            page_content['content'].extend(cleaned_paragraphs)

            # Extract tables using multiple methods
            tables = []

            # Try Camelot first
            camelot_tables = self._extract_tables_camelot(page_num + 1)
            tables.extend(camelot_tables)

            # Try pdfplumber as backup
            if not camelot_tables:
                plumber_tables = self._extract_tables_pdfplumber(plumber_page)
                tables.extend(plumber_tables)

            # Add tables to content (clean them first)
            for table in tables:
                raw = table.get('table_data', [])
                cleaned = self._clean_table(raw)
                if not cleaned:
                    continue
                table_content = {
                    'type': 'table',
                    'section': self._determine_section_context(table, hierarchy),
                    'description': f"Table extracted from page {page_num + 1}",
                    'table_data': cleaned
                }
                page_content['content'].append(table_content)

            # Extract charts/images
            charts = self._detect_charts_images(fitz_page, page_num + 1)
            for chart in charts:
                chart_content = {
                    'type': chart['type'],
                    'section': self._determine_section_context(chart, hierarchy),
                    'description': chart.get('description', ''),
                    'table_data': chart.get('table_data', [])
                }
                page_content['content'].append(chart_content)

        except Exception as e:
            self.logger.error(f"Error extracting content from page {page_num + 1}: {e}")

        return page_content

    def _extract_paragraphs(self, fitz_page, hierarchy: Dict) -> List[Dict[str, Any]]:
        """
        Extract paragraph content from a page.

        Args:
            fitz_page: PyMuPDF page object
            hierarchy: Section hierarchy information

        Returns:
            List of paragraph dictionaries
        """
        paragraphs = []

        try:
            # Get text blocks as list of (x0, y0, x1, y1, text)
            blocks = fitz_page.get_text("blocks")

            # Build a mapping of heading positions to text for better context assignment
            heading_positions = []
            for htext, info in hierarchy.items():
                heading_positions.append((htext, info.get('font_size', 0)))

            # Reflow adjacent short lines into paragraphs
            lines = []
            for block in blocks:
                raw = block[4]
                if not raw or not raw.strip():
                    continue
                # Skip candidate header/footer lines detected globally
                if hasattr(self, '_repeated_lines') and any(l in raw for l in self._repeated_lines):
                    continue
                # Split into physical lines and add
                for ln in raw.splitlines():
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)

            # Merge lines into paragraphs using simple heuristics
            merged = []
            buf = []
            for ln in lines:
                # If line looks like a heading (all caps or ends without punctuation and short), flush
                if self._is_likely_heading(ln):
                    if buf:
                        merged.append(' '.join(buf))
                        buf = []
                    merged.append(ln)
                    continue

                # Heuristic: if line ends with a hyphen, join without space
                if buf:
                    if buf[-1].endswith('-'):
                        buf[-1] = buf[-1][:-1] + ln
                    elif re.match(r'^[a-z0-9\)\]\%\,\.]', ln):
                        # starts like continuation, append with space
                        buf.append(ln)
                    else:
                        # If short line, likely still same paragraph
                        if len(ln) < 60:
                            buf.append(ln)
                        else:
                            buf.append(ln)
                else:
                    buf.append(ln)

            if buf:
                merged.append(' '.join(buf))

            # Now convert merged items into paragraph dicts, and assign simple section context
            current_section = None
            current_subsection = None
            for item in merged:
                if self._is_likely_heading(item):
                    # Treat as section / subsection based on casing and length
                    if item.isupper() or len(item) < 40:
                        current_section = item
                        current_subsection = None
                    else:
                        current_subsection = item
                    continue

                paragraph = {
                    'type': 'paragraph',
                    'section': current_section,
                    'sub_section': current_subsection,
                    'text': item
                }
                paragraphs.append(paragraph)

        except Exception as e:
            self.logger.warning(f"Error extracting paragraphs: {e}")

        return paragraphs

    def _determine_section_context(self, content_item: Dict, hierarchy: Dict) -> Optional[str]:
        """
        Determine the section context for a content item.

        Args:
            content_item: Content item (table, chart, etc.)
            hierarchy: Section hierarchy information

        Returns:
            Section name or None
        """
        # Improved heuristic: prefer the largest (level 1/2) heading if available
        if hierarchy:
            # hierarchy maps heading text -> info dict
            # prefer headings with smallest level number (1 is top)
            try:
                best = sorted(hierarchy.items(), key=lambda kv: kv[1].get('level', 99))[0][0]
                return best
            except Exception:
                return list(hierarchy.keys())[0]
        return None

    def extract_to_json(self) -> Dict[str, Any]:
        """
        Extract all content from PDF and return as structured JSON.

        Returns:
            Dictionary containing structured PDF content
        """
        result = {
            'document_info': {
                'filename': self.pdf_path.name,
                'total_pages': len(self.pymupdf_doc),
                'extraction_timestamp': pd.Timestamp.now().isoformat()
            },
            'pages': []
        }

        self.logger.info(f"Extracting content from {len(self.pymupdf_doc)} pages...")

        for page_num in range(len(self.pymupdf_doc)):
            self.logger.debug(f"Processing page {page_num + 1}")
            page_content = self._extract_page_content(page_num)
            result['pages'].append(page_content)

        return result

    def save_json(self, output_path: str, pretty: bool = True) -> None:
        """
        Extract content and save to JSON file.

        Args:
            output_path: Path to save JSON file
            pretty: Whether to format JSON with indentation
        """
        content = self.extract_to_json()

        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(content, f, indent=2, ensure_ascii=False)
            else:
                json.dump(content, f, ensure_ascii=False)

        self.logger.info(f"JSON output saved to: {output_path}")

    def close(self):
        """Close all open PDF documents."""
        if self.pymupdf_doc:
            self.pymupdf_doc.close()
        if self.pdfplumber_doc:
            self.pdfplumber_doc.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Main function to run the PDF parser from command line."""
    parser = argparse.ArgumentParser(
        description="Extract structured content from PDF files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_parser.py input.pdf output.json
  python pdf_parser.py input.pdf output.json --debug
  python pdf_parser.py input.pdf output.json --compact
        """
    )

    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('output_json', help='Path to output JSON file')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    parser.add_argument('--compact', action='store_true',
                       help='Generate compact JSON (no indentation)')

    args = parser.parse_args()

    # Validate input file — if missing, open a file-picker dialog for the user
    input_path = Path(args.input_pdf)
    if not input_path.exists():
        try:
            from tkinter import Tk, filedialog, messagebox
            root = Tk()
            root.withdraw()  # hide main window
            messagebox.showinfo(
                "PDF not found",
                f"Input file '{args.input_pdf}' was not found.\nPlease select a PDF file to parse."
            )
            selected = filedialog.askopenfilename(
                title="Select PDF file",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialdir=str(Path.home() / "Downloads")
            )
            root.destroy()
            if not selected:
                print("No file selected. Exiting.")
                return 1
            args.input_pdf = selected
            input_path = Path(args.input_pdf)
            print(f"Using selected file: {args.input_pdf}")
        except Exception as e:
            print(f"Error: Input file '{args.input_pdf}' does not exist and file dialog failed: {e}")
            return 1

    try:
        # Extract content
        with PDFContentExtractor(args.input_pdf, debug=args.debug) as extractor:
            extractor.save_json(args.output_json, pretty=not args.compact)

        print(f"Successfully extracted content from '{args.input_pdf}' to '{args.output_json}'")
        return 0

    except Exception as e:
        print(f"Error processing PDF: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
