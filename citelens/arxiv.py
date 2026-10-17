"""An async arXiv metadata client that deliberately goes slowly.

arXiv's API etiquette: at most one request every three seconds, over a single
connection. So this client limits *how many* requests are in flight (a
Semaphore) and *how often* a new one may start (a RateLimiter). Async doesn't
make arXiv answer faster. It lets the rest of the program, such as PDF
parsing, keep working while we wait our turn.
"""

from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from types import TracebackType
from typing import Self

import httpx
from pydantic import BaseModel, Field

from citelens.models import ArxivId, Author, Paper, is_arxiv_id

API_URL = "https://export.arxiv.org/api/query"
# From the arXiv API user manual: "no more than one request every three seconds".
ARXIV_INTERVAL_SECONDS = 3.0
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivMetadata(BaseModel):
    """What arXiv knows about a paper. Unlike the PDF heuristics, it's authoritative."""

    arxiv_id: ArxivId  # with version, e.g. "1706.03762v7"
    title: str = Field(min_length=1)
    authors: list[Author]
    abstract: str
    published: datetime
    categories: list[str]
    pdf_url: str


class PaperNotFound(LookupError):
    """arXiv answered, but has no paper with that ID."""


class RateLimiter:
    """Let a new request start at most once every ``interval`` seconds."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next_start = 0.0
        # Without the lock, two tasks could both read the same _next_start
        # and both start at once.
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_start:
                await asyncio.sleep(self._next_start - now)
                now = self._next_start
            self._next_start = now + self._interval


class ArxivClient:
    """Use as ``async with ArxivClient() as arxiv: await arxiv.fetch("1706.03762")``.

    ``transport`` exists so tests can swap the network for a fake one.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        interval: float = ARXIV_INTERVAL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._limiter = RateLimiter(interval)
        self._http = httpx.AsyncClient(
            transport=transport,
            timeout=30.0,
            headers={"User-Agent": "citelens/0.1 (research-paper assistant)"},
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def fetch(self, arxiv_id: str) -> ArxivMetadata:
        if not is_arxiv_id(arxiv_id):
            raise ValueError(f"not a valid arXiv ID: {arxiv_id!r}")
        async with self._semaphore:
            await self._limiter.wait()
            response = await self._http.get(
                API_URL, params={"id_list": arxiv_id, "max_results": 1}
            )
        response.raise_for_status()
        entries = parse_feed(response.text)
        if not entries:
            raise PaperNotFound(arxiv_id)
        return entries[0]

    async def fetch_many(self, arxiv_ids: Iterable[str]) -> list[ArxivMetadata]:
        """Fetch several papers. They're all *started* at once; the limits queue them."""
        return await asyncio.gather(*(self.fetch(arxiv_id) for arxiv_id in arxiv_ids))


def parse_feed(xml: str) -> list[ArxivMetadata]:
    """Turn an arXiv Atom feed into one ArxivMetadata per entry."""
    root = ET.fromstring(xml)
    results = []
    for entry in root.findall("atom:entry", _NS):
        entry_id = _text(entry, "atom:id")  # "http://arxiv.org/abs/1706.03762v7"
        if "/abs/" not in entry_id:
            continue  # an error entry, e.g. "http://arxiv.org/api/errors#..."
        pdf_links = [
            link.get("href", "")
            for link in entry.findall("atom:link", _NS)
            if link.get("title") == "pdf"
        ]
        results.append(
            ArxivMetadata(
                arxiv_id=entry_id.split("/abs/", 1)[1],
                title=_text(entry, "atom:title"),
                authors=[
                    Author(name=_text(author, "atom:name"))
                    for author in entry.findall("atom:author", _NS)
                ],
                abstract=_text(entry, "atom:summary"),
                published=datetime.fromisoformat(_text(entry, "atom:published")),
                categories=[
                    category.get("term", "")
                    for category in entry.findall("atom:category", _NS)
                ],
                pdf_url=pdf_links[0] if pdf_links else "",
            )
        )
    return results


def _text(element: ET.Element, path: str) -> str:
    """The child's text with line breaks and runs of spaces collapsed, as titles wrap in the feed."""
    child = element.find(path, _NS)
    if child is None or child.text is None:
        raise ValueError(f"arXiv entry is missing <{path}>")
    return " ".join(child.text.split())


def apply_metadata(paper: Paper, metadata: ArxivMetadata) -> Paper:
    """Trust arXiv over the PDF heuristics for title and authors."""
    return paper.model_copy(
        update={
            "title": metadata.title,
            "authors": metadata.authors,
            "arxiv_id": metadata.arxiv_id,
        }
    )
