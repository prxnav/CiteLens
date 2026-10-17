"""PDF opening and a lazy pages -> text blocks pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from citelens.timing import timed

_BOLD = 16  # PyMuPDF span flag bit for bold text
# Line direction; rotated text (e.g. the arXiv margin stamp) is skipped.
_HORIZONTAL = (1.0, 0.0)


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One block of text as laid out in the PDF, plus the font facts the parser needs."""

    page: int  # 1-based
    text: str
    size: float  # font size of the block's first span
    bold: bool  # True only if every span in the block is bold


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
def iter_text_blocks(pages: Iterator[pymupdf.Page]) -> Iterator[TextBlock]:
    """Lazily yield every non-empty, horizontal text block with its font size and boldness."""
    for page in pages:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # 1 = image block
                continue
            lines = [
                line for line in block["lines"] if tuple(line["dir"]) == _HORIZONTAL
            ]
            spans = [
                span for line in lines for span in line["spans"] if span["text"].strip()
            ]
            if not spans:
                continue
            text = join_lines(
                "".join(span["text"] for span in line["spans"]).strip()
                for line in lines
            )
            yield TextBlock(
                page=page.number + 1,
                text=text,
                size=round(spans[0]["size"], 1),
                bold=all(span["flags"] & _BOLD for span in spans),
            )


def join_lines(lines: Iterable[str]) -> str:
    """Join a block's lines with spaces, re-attaching words hyphenated across a line break."""
    text = ""
    for line in lines:
        if not line:
            continue
        if text.endswith("-") and line[0].islower():
            text = text[:-1] + line  # "representa-" + "tion" -> "representation"
        elif text:
            text += " " + line
        else:
            text = line
    return text
