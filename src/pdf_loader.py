"""
PDF Loader Module
=================
Extract text from scientific paper PDFs using PyMuPDF (fitz).
Supports single file and batch loading from a directory.
"""

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def load_pdf(path: str) -> str:
    """
    Load a single PDF file and extract all text content.

    Args:
        path: Path to the PDF file.

    Returns:
        Raw text extracted from all pages of the PDF.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If text extraction fails.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
        pages: List[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

        doc.close()
        return "\n\n".join(pages)

    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")


def load_all_pdfs(directory: str) -> Dict[str, str]:
    """
    Load all PDF files from a directory.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        Dictionary mapping filename -> extracted text.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {dir_path}")
        return {}

    papers: Dict[str, str] = {}
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        try:
            text = load_pdf(str(pdf_file))
            papers[pdf_file.name] = text
            print(f"    -> {len(text)} characters extracted")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    print(f"  Total: {len(papers)} papers loaded")
    return papers


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_file():
        text = load_pdf(str(target))
        print(f"Extracted {len(text)} characters from {target.name}")
        print(text[:500])
    elif target.is_dir():
        papers = load_all_pdfs(str(target))
        for name, text in papers.items():
            print(f"\n{name}: {len(text)} chars")
    else:
        print(f"Invalid path: {target}")
