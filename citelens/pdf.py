"""PDF opening and a lazy pages -> text blocks -> paragraphs pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pymupdf

from citelens.timing import timed


@contextmanager
def open_pdf(path: Path) -> Iterator[pymupdf.Document]:
    """Open a PDF and guarantee it closes, even if parsing raises partway through."""
    doc = pymupdf.open(path)
    try:
        yield doc
    finally:
        doc.close()


@timed("pages")
def iter_pages(doc: pymupdf.Document) -> Iterator[pymupdf.Page]:
    """Lazily yield each page of an already-open document."""
    for page_number in range(doc.page_count):
        yield doc[page_number]


@timed("text_blocks")
def iter_text_blocks(pages: Iterator[pymupdf.Page]) -> Iterator[tuple[int, str]]:
    """Lazily yield (page_number, block_text) for every non-empty text block."""
    for page in pages:
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if text:
                yield page.number + 1, text


@timed("paragraphs")
def iter_paragraphs(blocks: Iterator[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """Lazily split each text block into paragraphs on blank lines."""
    for page_number, block_text in blocks:
        for paragraph in block_text.split("\n\n"):
            stripped = paragraph.strip()
            if stripped:
                yield page_number, stripped
