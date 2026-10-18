"""Ingest arXiv PDFs: parse each file *while* fetching its metadata from arXiv.

The file name must be the arXiv ID, e.g. ``papers/1706.03762.pdf``.

Usage:
    uv run python scripts/ingest.py papers/*.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from citelens.arxiv import ArxivClient, apply_metadata
from citelens.models import Paper
from citelens.parser import parse_pdf


async def ingest(path: Path, arxiv: ArxivClient) -> Paper:
    arxiv_id = path.stem
    data = path.read_bytes()
    # parse_pdf is ordinary blocking code, so it runs in a worker thread.
    # Calling it directly here would freeze the event loop, so nothing else
    # (e.g. the arXiv request) could make progress until parsing finished.
    paper, metadata = await asyncio.gather(
        asyncio.to_thread(parse_pdf, data, fallback_title=arxiv_id, arxiv_id=arxiv_id),
        arxiv.fetch(arxiv_id),
    )
    return apply_metadata(paper, metadata)


async def main(paths: list[Path]) -> None:
    start = time.perf_counter()
    async with ArxivClient() as arxiv:

        async def ingest_and_report(path: Path) -> None:
            paper = await ingest(path, arxiv)
            authors = ", ".join(author.name for author in paper.authors[:3])
            print(
                f"\n{time.perf_counter() - start:5.1f}s  {paper.arxiv_id}  {paper.title}\n"
                f"       {authors}{' et al.' if len(paper.authors) > 3 else ''}\n"
                f"       {len(paper.sections)} sections, {len(paper.chunks)} chunks, "
                f"{len(paper.references)} references"
            )

        await asyncio.gather(*(ingest_and_report(path) for path in paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", type=Path, nargs="+", help="PDFs named <arxiv-id>.pdf")
    asyncio.run(main(parser.parse_args().pdfs))
