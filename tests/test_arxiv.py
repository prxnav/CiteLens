import asyncio
import itertools
import time
from datetime import UTC, datetime

import httpx
import pytest

from citelens.arxiv import ArxivClient, PaperNotFound, apply_metadata, parse_feed
from citelens.models import Paper

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All
      You Need</title>
    <link href="https://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/1706.03762v7" rel="related" title="pdf"/>
    <summary>The dominant sequence transduction models...</summary>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <published>2017-06-12T17:57:34Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""

EMPTY_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


def fake_arxiv(handler: httpx.MockTransport) -> ArxivClient:
    """A client whose requests go to ``handler`` instead of the network, with no waiting."""
    return ArxivClient(interval=0.0, transport=handler)


def test_parse_feed_reads_an_entry() -> None:
    [metadata] = parse_feed(FEED)
    assert metadata.arxiv_id == "1706.03762v7"
    assert metadata.title == "Attention Is All You Need"  # line break collapsed
    assert [author.name for author in metadata.authors] == [
        "Ashish Vaswani",
        "Noam Shazeer",
    ]
    assert metadata.published == datetime(2017, 6, 12, 17, 57, 34, tzinfo=UTC)
    assert metadata.categories == ["cs.CL", "cs.LG"]
    assert metadata.pdf_url == "https://arxiv.org/pdf/1706.03762v7"


def test_fetch_sends_the_id_and_parses_the_answer() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, text=FEED)

    async def run() -> str:
        async with fake_arxiv(httpx.MockTransport(handler)) as arxiv:
            return (await arxiv.fetch("1706.03762")).title

    assert asyncio.run(run()) == "Attention Is All You Need"
    assert seen[0].params["id_list"] == "1706.03762"


def test_fetch_raises_when_arxiv_has_no_such_paper() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=EMPTY_FEED)
    )

    async def run() -> None:
        async with fake_arxiv(transport) as arxiv:
            await arxiv.fetch("9999.99999")

    with pytest.raises(PaperNotFound):
        asyncio.run(run())


def test_fetch_rejects_a_bad_id_without_a_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    async def run() -> None:
        async with fake_arxiv(httpx.MockTransport(handler)) as arxiv:
            await arxiv.fetch("not-an-id")

    with pytest.raises(ValueError):
        asyncio.run(run())


def test_semaphore_caps_requests_in_flight() -> None:
    in_flight = 0
    most_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, most_in_flight
        in_flight += 1
        most_in_flight = max(most_in_flight, in_flight)
        await asyncio.sleep(0.01)  # a slow server, so requests could overlap
        in_flight -= 1
        return httpx.Response(200, text=FEED)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with ArxivClient(
            max_concurrent=2, interval=0.0, transport=transport
        ) as arxiv:
            await arxiv.fetch_many(["1706.03762"] * 6)

    asyncio.run(run())
    assert most_in_flight == 2


def test_rate_limiter_spaces_out_request_starts() -> None:
    interval = 0.05
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(time.monotonic())
        return httpx.Response(200, text=FEED)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        # Plenty of concurrency allowed, so only the rate limit holds requests back.
        async with ArxivClient(
            max_concurrent=10, interval=interval, transport=transport
        ) as arxiv:
            await arxiv.fetch_many(["1706.03762"] * 4)

    asyncio.run(run())
    gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
    assert all(gap >= interval * 0.9 for gap in gaps), gaps


def test_apply_metadata_trusts_arxiv_for_title_and_authors() -> None:
    [metadata] = parse_feed(FEED)
    paper = apply_metadata(Paper(title="guessed from font size"), metadata)
    assert paper.title == "Attention Is All You Need"
    assert [author.name for author in paper.authors] == [
        "Ashish Vaswani",
        "Noam Shazeer",
    ]
    assert paper.arxiv_id == "1706.03762v7"
