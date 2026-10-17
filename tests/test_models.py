import pytest
from pydantic import ValidationError

from citelens.models import Chunk, Paper, Reference


def test_valid_chunk() -> None:
    chunk = Chunk(text="We introduce BERT", page=1, char_start=100, char_end=117)
    assert chunk.section_path == []


def test_page_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Chunk(text="x", page=0, char_start=0, char_end=1)


def test_span_must_match_text_length() -> None:
    with pytest.raises(ValidationError, match="span 0:5"):
        Chunk(text="We introduce BERT", page=1, char_start=0, char_end=5)


@pytest.mark.parametrize(
    "arxiv_id", ["1810.04805", "1810.04805v2", "0704.0001", "hep-th/9901001"]
)
def test_accepts_real_arxiv_ids(arxiv_id: str) -> None:
    assert Paper(title="t", arxiv_id=arxiv_id).arxiv_id == arxiv_id


@pytest.mark.parametrize(
    "arxiv_id", ["hello", "1810.048", "https://arxiv.org/abs/1810.04805"]
)
def test_rejects_bad_arxiv_ids(arxiv_id: str) -> None:
    with pytest.raises(ValidationError, match="not a valid arXiv ID"):
        Paper(title="t", arxiv_id=arxiv_id)


def test_paper_rejects_chunk_span_not_in_full_text() -> None:
    chunk = Chunk(text="BERT", page=1, char_start=0, char_end=4)
    with pytest.raises(ValidationError, match="doesn't match"):
        Paper(title="t", chunks=[chunk], full_text="ELMo and BERT")


def test_reference_arxiv_id_is_validated_too() -> None:
    with pytest.raises(ValidationError):
        Reference(raw="[1] Someone. 2020.", arxiv_id="nope")
