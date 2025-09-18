#!/usr/bin/env python3
"""
Setup script for PDF Parser and Structured JSON Extractor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove version constraints for basic setup
            req = line.split('>=')[0].split('==')[0].split('[')[0]
            requirements.append(line)

setup(
    name="pdf-parser-json-extractor",
    version="1.0.0",
    author="Assignment Solution",
    author_email="example@email.com",
    description="A comprehensive Python program for parsing PDFs and extracting structured JSON",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/pdf-parser",
    py_modules=["pdf_parser"],
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pdf parsing json extraction table chart document",
    entry_points={
        "console_scripts": [
            "pdf-parser=pdf_parser:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
