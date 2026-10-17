"""A naive parser: PDF text blocks -> Paper, guessing structure from font size and boldness.

It *will* be wrong on some papers. That's expected — Phase 02 replaces it with
GROBID and measures the difference.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from citelens.models import Chunk, Paper, Reference, Section, is_arxiv_id
from citelens.pdf import TextBlock, iter_pages, iter_text_blocks, join_lines, open_pdf

# "1 Introduction", "3.1 Pre-training BERT", "3.1. Residual Learning", "A Appendix Title"
_NUMBERED_HEADING = re.compile(r"^((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+[A-Z]")
_UNNUMBERED_HEADINGS = {
    "abstract",
    "references",
    "acknowledgements",
    "acknowledgments",
    "appendix",
}
_MAX_HEADING_LENGTH = 80
# Body text in one paper can wobble between e.g. 9.9, 10.0 and 10.1.
_SIZE_TOLERANCE = 0.5
_SENTENCE_END = (".", "!", "?", ":")
_LEADING_YEAR = re.compile(r"^(19|20)\d{2}\b")
# Zero-width match, so splitting on it cuts *before* each "[12] " without consuming it.
_BEFORE_REFERENCE_NUMBER = re.compile(r"(?=\[\d+\]\s)")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_IN_TEXT = re.compile(r"arXiv:\s?(\d{4}\.\d{4,5})")
_CHUNK_SEPARATOR = "\n\n"


@dataclass
class _DraftChunk:
    page: int
    section_path: list[str]
    text: str


def parse_pdf(path: Path) -> Paper:
    """Parse a PDF file into a Paper. Files named like ``1810.04805.pdf`` get that arXiv ID."""
    with open_pdf(path) as doc:
        blocks = list(iter_text_blocks(iter_pages(doc)))
    arxiv_id = path.stem if is_arxiv_id(path.stem) else None
    return build_paper(blocks, fallback_title=path.stem, arxiv_id=arxiv_id)


def build_paper(
    blocks: list[TextBlock], *, fallback_title: str, arxiv_id: str | None = None
) -> Paper:
    """Turn already-extracted blocks into a Paper. Separate from parse_pdf so tests need no PDF."""
    body_size = _body_font_size(blocks)
    title_indexes = _title_block_indexes(blocks)

    sections: list[Section] = []
    heading_stack: list[Section] = []  # the headings the current block sits under
    drafts: list[_DraftChunk] = []
    reference_texts: list[str] = []
    in_references = False

    for index, block in enumerate(blocks):
        # Skip the title (stored separately) and bare page numbers.
        if index in title_indexes or block.text.isdigit():
            continue

        level = _heading_level(block, body_size)
        if level is not None:
            section = Section(title=block.text, level=level, page=block.page)
            sections.append(section)
            while heading_stack and heading_stack[-1].level >= level:
                heading_stack.pop()
            heading_stack.append(section)
            in_references = block.text.lower() == "references"
            continue

        if in_references:
            reference_texts.append(block.text)
        else:
            _add_chunk_text(drafts, block, [section.title for section in heading_stack])

    title = (
        " ".join(blocks[index].text for index in sorted(title_indexes))
        or fallback_title
    )
    chunks, full_text = _place_chunks(drafts)
    return Paper(
        title=title,
        arxiv_id=arxiv_id,
        sections=sections,
        chunks=chunks,
        references=[_to_reference(raw) for raw in _split_references(reference_texts)],
        full_text=full_text,
    )


def _body_font_size(blocks: list[TextBlock]) -> float:
    """The font size covering the most characters — almost always the body text."""
    sizes: Counter[float] = Counter()
    for block in blocks:
        sizes[block.size] += len(block.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _title_block_indexes(blocks: list[TextBlock]) -> set[int]:
    """The title is the largest text on page 1, possibly wrapped over several blocks."""
    first_page = [
        (index, block) for index, block in enumerate(blocks) if block.page == 1
    ]
    if not first_page:
        return set()
    largest = max(block.size for _, block in first_page)
    return {index for index, block in first_page if block.size == largest}


def _heading_level(block: TextBlock, body_size: float) -> int | None:
    """Return 1 for "3 BERT", 2 for "3.1 Pre-training BERT", ...; None if not a heading.

    A heading must be entirely bold, at least body size (which rules out bold
    numbers in smaller table text), short, and either numbered or a known
    unnumbered heading like "Abstract".
    """
    if (
        not block.bold
        or block.size < body_size - _SIZE_TOLERANCE
        or len(block.text) > _MAX_HEADING_LENGTH
    ):
        return None
    if block.text.lower() in _UNNUMBERED_HEADINGS:
        return 1
    match = _NUMBERED_HEADING.match(block.text)
    if match is None:
        return None
    return match.group(1).count(".") + 1


def _add_chunk_text(
    drafts: list[_DraftChunk], block: TextBlock, section_path: list[str]
) -> None:
    """Start a new chunk, or glue this block onto the last one if it looks like a mid-sentence split.

    PDFs often break one paragraph into several blocks (at a column or page
    boundary). If the previous chunk doesn't end a sentence and this block
    starts lowercase, they're very likely the same paragraph.
    """
    previous = drafts[-1] if drafts else None
    if (
        previous is not None
        and previous.section_path == section_path
        and not previous.text.endswith(_SENTENCE_END)
        and block.text[0].islower()
    ):
        previous.text = join_lines([previous.text, block.text])
    else:
        drafts.append(
            _DraftChunk(page=block.page, section_path=section_path, text=block.text)
        )


def _split_references(texts: list[str]) -> list[str]:
    """Regroup the reference section's blocks into one string per bibliography entry.

    Block boundaries don't line up with entries, so each citation style needs its own rule.
    """
    if texts and texts[0].startswith("["):
        # Numbered style ("[12] ..."): blocks can split mid-entry *and* hold several
        # entries, so join everything and cut before each "[n]".
        stream = join_lines(texts)
        return [
            entry.strip()
            for entry in _BEFORE_REFERENCE_NUMBER.split(stream)
            if entry.strip()
        ]

    # Author-year style: an entry ends with "." and the next one starts with a name,
    # so a block that starts lowercase or with the year belongs to the previous entry.
    entries: list[str] = []
    for text in texts:
        previous = entries[-1] if entries else None
        if previous is not None and (
            not previous.endswith(".") or text[0].islower() or _LEADING_YEAR.match(text)
        ):
            entries[-1] = join_lines([previous, text])
        else:
            entries.append(text)
    return entries


def _to_reference(raw: str) -> Reference:
    year = _YEAR.search(raw)
    arxiv = _ARXIV_IN_TEXT.search(raw)
    return Reference(
        raw=raw,
        year=int(year.group(0)) if year else None,
        arxiv_id=arxiv.group(1) if arxiv else None,
    )


def _place_chunks(drafts: list[_DraftChunk]) -> tuple[list[Chunk], str]:
    """Lay the chunks end to end into one full text, recording each one's exact span."""
    chunks: list[Chunk] = []
    position = 0
    for draft in drafts:
        chunks.append(
            Chunk(
                text=draft.text,
                page=draft.page,
                section_path=draft.section_path,
                char_start=position,
                char_end=position + len(draft.text),
            )
        )
        position += len(draft.text) + len(_CHUNK_SEPARATOR)
    return chunks, _CHUNK_SEPARATOR.join(draft.text for draft in drafts)
