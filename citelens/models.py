"""The citelens domain: what a parsed paper is made of."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

# New-style IDs ("1810.04805", "1810.04805v2") and old-style ones
# ("hep-th/9901001") — arXiv switched formats in April 2007.
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")


def is_arxiv_id(value: str) -> bool:
    return _ARXIV_ID.match(value) is not None


def _check_arxiv_id(value: str) -> str:
    if not is_arxiv_id(value):
        raise ValueError(f"not a valid arXiv ID: {value!r}")
    return value


ArxivId = Annotated[str, AfterValidator(_check_arxiv_id)]


class Author(BaseModel):
    name: str = Field(min_length=1)
    affiliation: str | None = None


class Section(BaseModel):
    """A titled part of the paper, e.g. "3.1 Pre-training BERT"."""

    title: str = Field(min_length=1)
    level: int = Field(ge=1)  # 1 = "3 BERT", 2 = "3.1 Pre-training BERT", ...
    page: int = Field(ge=1)


class Chunk(BaseModel):
    """A piece of text plus exactly where it came from.

    ``char_start``/``char_end`` are offsets into the paper's full extracted
    text, so ``full_text[char_start:char_end] == text``. That exactness is
    what lets a later citation point at — and be checked against — the real
    source span, not just a page.
    """

    text: str = Field(min_length=1)
    page: int = Field(ge=1)
    # e.g. ["3 BERT", "3.1 Pre-training BERT"]
    section_path: list[str] = Field(default_factory=list)
    char_start: int = Field(ge=0)
    char_end: int

    @model_validator(mode="after")
    def _span_matches_text(self) -> Chunk:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(
                f"span {self.char_start}:{self.char_end} has length "
                f"{self.char_end - self.char_start}, but text has length {len(self.text)}"
            )
        return self


class Reference(BaseModel):
    """One bibliography entry. Only ``raw`` is guaranteed; the rest is best-effort."""

    raw: str = Field(min_length=1)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    arxiv_id: ArxivId | None = None


class Paper(BaseModel):
    title: str = Field(min_length=1)
    arxiv_id: ArxivId | None = None  # None for PDFs that didn't come from arXiv
    authors: list[Author] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    full_text: str = ""  # what every Chunk's char_start/char_end point into

    @model_validator(mode="after")
    def _chunks_point_into_full_text(self) -> Paper:
        for chunk in self.chunks:
            if self.full_text[chunk.char_start : chunk.char_end] != chunk.text:
                raise ValueError(
                    f"chunk span {chunk.char_start}:{chunk.char_end} doesn't match its text in full_text"
                )
        return self
